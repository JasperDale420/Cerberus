#!/usr/bin/env python
"""Cerberus Autoresearch Evaluation Runner — FROZEN, do not modify.

Runs a speed-optimized WFO for a given strategy and outputs parseable metrics.
Used by the autoresearch loop (program_cerberus.md) to evaluate strategy changes.

Design principles:
- Full regime diversity (2021-2025) to prevent overfitting to a single market
- Reduced trials/symbols/windows for ~15-25 min iterations
- Per-window regime tagging for identifying regime-specific strengths
- Dynamic strategy import so new strategies don't need registry changes

Usage:
    uv run python scripts/cerberus_autoresearch.py <strategy_name> [--n-trials N] [--n-symbols N]
"""

import importlib.util
import json
import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress all logging before any imports
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["EMPIRE_LOG_LEVEL"] = "CRITICAL"
os.environ.setdefault("EMPIRE_LOG_FORMAT", "json")

os.chdir("/Users/jacobmcmillan/Empire/Cerberus")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.analytics.optuna_harness import WalkForwardOptimizer  # noqa: E402

# ── Fixed configuration ──────────────────────────────────────────────

# Default symbol universe: diversified, liquid, covers multiple sectors
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "META"]

# WFO parameters tuned for speed with regime diversity
WFO_FULL_START = "2021-01-01"
WFO_FULL_END = "2025-09-30"
WFO_TRAIN_MONTHS = 8
WFO_TEST_MONTHS = 3
WFO_HOLDOUT_MONTHS = 3
WFO_MODE = "rolling"
DATA_DIR = "data/bars_2023_2025"
CONFIG_PATH = "config/backtest_v2.yaml"


def classify_window_regime(data_dir: str, start: str, end: str) -> str:
    """Classify the dominant trend/vol regime for an OOS window using SPY data."""
    spy_path = Path(data_dir) / "SPY_1Min.parquet"
    if not spy_path.exists():
        return "unknown"

    try:
        df = pd.read_parquet(spy_path, columns=["timestamp", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        mask = (df["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (df["timestamp"] <= pd.Timestamp(end, tz="UTC"))
        window_df = df.loc[mask]

        if len(window_df) < 100:
            return "insufficient_data"

        # Daily OHLC for regime classification
        daily = window_df.set_index("timestamp")["close"].resample("1D").agg(["first", "last", "max", "min"])
        daily = daily.dropna()

        if len(daily) < 5:
            return "insufficient_data"

        # Trend: compare first vs last close
        total_return = (daily["last"].iloc[-1] / daily["first"].iloc[0]) - 1
        if total_return > 0.03:
            trend = "trending_up"
        elif total_return < -0.03:
            trend = "trending_down"
        else:
            trend = "choppy"

        # Volatility: annualized daily return std
        daily_returns = daily["last"].pct_change().dropna()
        ann_vol = daily_returns.std() * np.sqrt(252)
        if ann_vol > 0.25:
            vol = "high_vol"
        elif ann_vol < 0.12:
            vol = "low_vol"
        else:
            vol = "normal_vol"

        return f"{trend}+{vol}"
    except Exception:
        return "classification_error"


def dynamic_import_strategy(strategy_name: str) -> bool:
    """Try to import a strategy module dynamically if not in the standard registry."""
    strategy_path = Path(f"src/strategies/{strategy_name}.py")
    if not strategy_path.exists():
        return False

    try:
        spec = importlib.util.spec_from_file_location(
            f"src.strategies.{strategy_name}",
            strategy_path,
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return True
    except Exception as e:
        print(f"AUTORESEARCH_ERROR dynamic_import_failed strategy={strategy_name} error={e}", flush=True)
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cerberus Autoresearch Evaluation Runner")
    parser.add_argument("strategy", help="Strategy name to evaluate")
    parser.add_argument("--n-trials", type=int, default=8, help="Optuna trials per window")
    parser.add_argument("--n-symbols", type=int, default=8, help="Number of symbols")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Bar data directory")
    args = parser.parse_args()

    strategy_name = args.strategy

    # Attempt dynamic import for new strategies
    dynamic_import_strategy(strategy_name)

    # Load base config
    with open(CONFIG_PATH) as f:
        base_config = yaml.safe_load(f)

    # Configure universe
    symbols = DEFAULT_SYMBOLS[: args.n_symbols]
    base_config["universe"] = {"symbols": symbols}
    base_config["database_url"] = "sqlite://"
    base_config["log_level"] = "CRITICAL"

    # Enable ONLY the target strategy
    for s in base_config.get("strategies", {}):
        base_config["strategies"][s]["enabled"] = s == strategy_name

    # If strategy not in config, add a minimal entry
    if strategy_name not in base_config.get("strategies", {}):
        base_config.setdefault("strategies", {})[strategy_name] = {"enabled": True}

    # Build and run WFO
    wfo = WalkForwardOptimizer(
        full_start=WFO_FULL_START,
        full_end=WFO_FULL_END,
        min_train_months=WFO_TRAIN_MONTHS,
        test_months=WFO_TEST_MONTHS,
        n_trials=args.n_trials,
        holdout_months=WFO_HOLDOUT_MONTHS,
        mode=WFO_MODE,
    )

    windows = wfo.get_windows()

    print(
        f"AUTORESEARCH_START strategy={strategy_name} windows={len(windows)} "
        f"trials={args.n_trials} symbols={len(symbols)}",
        flush=True,
    )

    try:
        results = wfo.run(
            strategy_name=strategy_name,
            base_config=base_config,
            data_dir=args.data_dir,
            config_path=CONFIG_PATH,
            workers=2,
        )
    except Exception as e:
        print(f"AUTORESEARCH_ERROR wfo_failed strategy={strategy_name} error={e}", flush=True)
        sys.exit(1)

    # ── Parse results ──────────────────────────────────────────────
    oos_scores = results.get("oos_scores", [])
    oos_metrics = results.get("oos_metrics", [])
    param_stability = results.get("param_stability", {})

    positive_windows = sum(1 for s in oos_scores if s > 0)
    total_windows = len(oos_scores)
    total_oos_trades = sum(m.get("n_trades", 0) for m in oos_metrics)

    # Avg Sortino (only from windows with trades)
    sortinos = [m.get("sortino_ratio", 0) for m in oos_metrics if m.get("n_trades", 0) > 0]
    avg_sortino = sum(sortinos) / max(len(sortinos), 1)

    # Composite score: mean of OOS scores (excluding hard-reject sentinel values)
    valid_scores = [s for s in oos_scores if s > -100]
    composite_score = sum(valid_scores) / max(len(valid_scores), 1) if valid_scores else -999.0

    # Max param CV
    param_cvs = [stats.get("cv", 0.0) for stats in param_stability.values()]
    param_cv_max = max(param_cvs) if param_cvs else 0.0

    # WFO efficiency
    wfo_efficiency = results.get("wfo_efficiency_ratio", 0.0)

    # ── Per-window regime breakdown ─────────────────────────────────
    for i, window in enumerate(windows):
        regime = classify_window_regime(args.data_dir, window["test_start"], window["test_end"])
        m = oos_metrics[i] if i < len(oos_metrics) else {}
        score = oos_scores[i] if i < len(oos_scores) else -999.0
        print(
            f"REGIME_BREAKDOWN window={i} regime={regime} "
            f"oos_score={score:.4f} trades={m.get('n_trades', 0)} "
            f"pf={m.get('profit_factor', 0.0):.2f} sharpe={m.get('sharpe_ratio', 0.0):.3f}",
            flush=True,
        )

    # ── Summary result line ─────────────────────────────────────────
    print(
        f"AUTORESEARCH_RESULT strategy={strategy_name} "
        f"composite_score={composite_score:.4f} "
        f"windows_profitable={positive_windows}/{total_windows} "
        f"total_oos_trades={total_oos_trades} "
        f"avg_sortino={avg_sortino:.4f} "
        f"param_cv_max={param_cv_max:.4f} "
        f"wfo_efficiency={wfo_efficiency:.4f}",
        flush=True,
    )

    # ── Save full results JSON ─────────────────────────────────────
    out_dir = "artifacts/autoresearch"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{strategy_name}_latest.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Exit 0 = success
    sys.exit(0)


if __name__ == "__main__":
    main()
