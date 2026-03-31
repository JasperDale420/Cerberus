#!/usr/bin/env python
"""Cerberus Autoresearch Evaluation Runner — FROZEN, do not modify.

Runs a speed-optimized WFO for a given strategy and outputs parseable metrics.
Used by the autoresearch driver (scripts/autoresearch_driver.sh) to evaluate
strategy changes. Verbose Optuna/WFO output goes to a log file; only parseable
summary lines go to stdout.

Design principles:
- Full regime diversity (2022-2025) to prevent overfitting to a single market
- Reduced trials/symbols/windows for ~15-25 min iterations
- Per-window regime tagging for identifying regime-specific strengths
- Dynamic strategy import so new strategies don't need registry changes

Usage:
    uv run python scripts/cerberus_autoresearch.py <strategy_name> [--n-trials N] [--n-symbols N] [--log-dir DIR]
"""

import importlib.util
import json
import logging
import os
import sys
import warnings
from datetime import datetime
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
# 5 windows x 6-month OOS = ~15-20 min per iteration
# Wider windows = fewer windows but more trades per OOS (statistically meaningful)
# Covers: 2021 bull, 2022 bear, 2023 recovery, 2024 bull, early 2025
WFO_FULL_START = "2020-06-01"
WFO_FULL_END = "2025-09-30"
WFO_TRAIN_MONTHS = 12
WFO_TEST_MONTHS = 6
WFO_HOLDOUT_MONTHS = 3
WFO_MODE = "rolling"
DATA_DIR = "data/bars_2023_2025"
CONFIG_PATH = "config/backtest_v2.yaml"


def classify_window_regime(data_dir: str, start: str, end: str) -> str:
    """Classify the dominant trend/vol regime for an OOS window using optimized detector.

    Uses the grid-search-optimized 2-axis detector (81.9% accuracy):
    - Trend: Dual SMA crossover (fast=10, slow=40, flat_band=1%)
    - Vol: 30-day realized vol (LOW<8%, HIGH>20%, SHOCK>50%)

    If pre-labeled regime data exists in data/regime_labeled/, uses that directly.
    Otherwise falls back to computing from raw 1-minute bars.
    """
    # Try pre-labeled data first (fast path)
    regime_dir = Path("data/regime_labeled")
    spy_regime = regime_dir / "SPY_daily_regime.parquet"
    if spy_regime.exists():
        try:
            df = pd.read_parquet(spy_regime)
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start) & (df["date"] <= end)
            window = df.loc[mask]
            if len(window) >= 5:
                # Return the dominant regime combination
                regime_counts = window["regime"].value_counts()
                return str(regime_counts.index[0])
        except Exception:
            pass  # Fall through to raw computation

    # Fallback: compute from raw bars
    spy_path = Path(data_dir) / "SPY_1Min.parquet"
    if not spy_path.exists():
        return "unknown"

    try:
        df = pd.read_parquet(spy_path, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        mask = (df["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (df["timestamp"] <= pd.Timestamp(end, tz="UTC"))
        window_df = df.loc[mask]

        if len(window_df) < 100:
            return "insufficient_data"

        # Resample to daily
        daily = window_df.set_index("timestamp")["close"].resample("1D").last().dropna()
        if len(daily) < 40:
            return "insufficient_data"

        closes = daily.values.astype(float)

        # Trend: Dual SMA crossover (optimized: fast=10, slow=40, band=1%)
        sma_fast = pd.Series(closes).rolling(10).mean().values
        sma_slow = pd.Series(closes).rolling(40).mean().values
        last = len(closes) - 1
        if np.isnan(sma_slow[last]):
            trend = "FLAT"
        else:
            pct_above = (closes[last] - sma_slow[last]) / sma_slow[last]
            fast_above = (sma_fast[last] - sma_slow[last]) / sma_slow[last]
            if pct_above > 0.01 and fast_above > 0:
                trend = "UP"
            elif pct_above < -0.01 and fast_above < 0:
                trend = "DOWN"
            else:
                trend = "FLAT"

        # Vol: 30-day realized vol (optimized: LOW<8%, HIGH>20%, SHOCK>50%)
        log_rets = np.diff(np.log(closes))
        rvol = float(np.std(log_rets[-30:]) * np.sqrt(252)) if len(log_rets) >= 30 else 0.15
        if rvol >= 0.50:
            vol = "SHOCK"
        elif rvol >= 0.20:
            vol = "HIGH"
        elif rvol <= 0.08:
            vol = "LOW"
        else:
            vol = "NORMAL"

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
        sys.stderr.write(f"AUTORESEARCH_ERROR dynamic_import_failed strategy={strategy_name} error={e}\n")
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cerberus Autoresearch Evaluation Runner")
    parser.add_argument("strategy", help="Strategy name to evaluate")
    parser.add_argument("--n-trials", type=int, default=8, help="Optuna trials per window")
    parser.add_argument("--n-symbols", type=int, default=8, help="Number of symbols")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Bar data directory")
    parser.add_argument("--log-dir", default="artifacts/autoresearch/logs", help="Directory for verbose WFO logs")
    parser.add_argument(
        "--target-regime",
        default=None,
        help="Score only windows matching this regime (e.g., UP+NORMAL). Others excluded from composite.",
    )
    args = parser.parse_args()

    strategy_name = args.strategy

    # Redirect stdout/stderr to log file for verbose WFO output.
    # We keep a reference to real stdout for our parseable summary lines.
    real_stdout = sys.stdout
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{strategy_name}_{ts}.log"
    log_file = open(log_path, "w")  # noqa: SIM115
    sys.stdout = log_file
    sys.stderr = log_file

    def emit(msg: str) -> None:
        """Print to both real stdout (for driver parsing) and log file."""
        real_stdout.write(msg + "\n")
        real_stdout.flush()
        log_file.write(msg + "\n")
        log_file.flush()

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

    emit(
        f"AUTORESEARCH_START strategy={strategy_name} windows={len(windows)} "
        f"trials={args.n_trials} symbols={len(symbols)}"
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
        emit(f"AUTORESEARCH_ERROR wfo_failed strategy={strategy_name} error={e}")
        log_file.close()
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

    # Simplicity bonus/penalty based on strategy LOC (Karpathy: simpler is better)
    strategy_path = Path(f"src/strategies/{strategy_name}.py")
    strategy_loc = 0
    if strategy_path.exists():
        lines = strategy_path.read_text().splitlines()
        strategy_loc = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))
    if strategy_loc > 100:
        loc_penalty = max(-2.0, -0.05 * (strategy_loc - 100))  # -0.05/line, capped at -2.0
    elif strategy_loc < 50 and strategy_loc > 0:
        loc_penalty = min(0.5, 0.02 * (50 - strategy_loc))  # +0.02/line, capped at +0.5
    else:
        loc_penalty = 0.0
    if composite_score > -100:  # Only apply to real scores, not sentinel -999
        composite_score += loc_penalty

    # Max param CV
    param_cvs = [stats.get("cv", 0.0) for stats in param_stability.values()]
    param_cv_max = max(param_cvs) if param_cvs else 0.0

    # WFO efficiency
    wfo_efficiency = results.get("wfo_efficiency_ratio", 0.0)

    # ── Per-window regime breakdown ─────────────────────────────────
    regime_stats: dict[str, list[dict]] = {}
    for i, window in enumerate(windows):
        regime = classify_window_regime(args.data_dir, window["test_start"], window["test_end"])
        m = oos_metrics[i] if i < len(oos_metrics) else {}
        score = oos_scores[i] if i < len(oos_scores) else -999.0
        emit(
            f"REGIME_BREAKDOWN window={i} regime={regime} "
            f"oos_score={score:.4f} trades={m.get('n_trades', 0)} "
            f"pf={m.get('profit_factor', 0.0):.2f} sharpe={m.get('sharpe_ratio', 0.0):.3f}"
        )
        regime_stats.setdefault(regime, []).append(
            {
                "score": score,
                "trades": m.get("n_trades", 0),
                "pf": m.get("profit_factor", 0.0),
                "sharpe": m.get("sharpe_ratio", 0.0),
                "sortino": m.get("sortino_ratio", 0.0),
                "pnl": m.get("net_pnl", 0.0),
            }
        )

    # ── Per-regime aggregate ───────────────────────────────────────
    best_regime = ""
    best_regime_pf = 0.0
    worst_regime = ""
    worst_regime_pf = 999.0
    for regime, stats_list in sorted(regime_stats.items()):
        wt = [s for s in stats_list if s["trades"] > 0]
        if not wt:
            emit(f"REGIME_AGGREGATE regime={regime} windows={len(stats_list)} trades=0 avg_pf=0.00 avg_sharpe=0.000")
            continue
        total_trades = sum(s["trades"] for s in wt)
        avg_pf_r = np.mean([s["pf"] for s in wt])
        avg_sharpe_r = np.mean([s["sharpe"] for s in wt])
        avg_sortino_r = np.mean([s["sortino"] for s in wt])
        total_pnl = sum(s["pnl"] for s in wt)
        profitable = sum(1 for s in wt if s["pf"] > 1.0)
        emit(
            f"REGIME_AGGREGATE regime={regime} windows={len(stats_list)} "
            f"trades={total_trades} avg_pf={avg_pf_r:.2f} avg_sharpe={avg_sharpe_r:.3f} "
            f"avg_sortino={avg_sortino_r:.3f} total_pnl={total_pnl:.2f} "
            f"profitable={profitable}/{len(wt)}"
        )
        if avg_pf_r > best_regime_pf:
            best_regime_pf = avg_pf_r
            best_regime = regime
        if avg_pf_r < worst_regime_pf:
            worst_regime_pf = avg_pf_r
            worst_regime = regime

    # ── Regime-filtered scoring for specialists ─────────────────────
    # When --target-regime is set, recompute composite using only matching windows.
    # This prevents a UP+NORMAL specialist from being penalized for DOWN+HIGH windows.
    target_regime = args.target_regime
    if target_regime and regime_stats.get(target_regime):
        target_windows = regime_stats[target_regime]
        target_valid = [s["score"] for s in target_windows if s["score"] > -100]
        if target_valid:
            composite_score = sum(target_valid) / len(target_valid)
        target_trades = sum(s["trades"] for s in target_windows)
        target_positive = sum(1 for s in target_windows if s["score"] > 0)
        emit(
            f"REGIME_FILTERED target={target_regime} "
            f"composite_score={composite_score:.4f} "
            f"windows={target_positive}/{len(target_windows)} "
            f"trades={target_trades}"
        )
        # Override totals for the summary line
        positive_windows = target_positive
        total_windows = len(target_windows)
        total_oos_trades = target_trades

    # ── Summary result line ─────────────────────────────────────────
    emit(
        f"AUTORESEARCH_RESULT strategy={strategy_name} "
        f"composite_score={composite_score:.4f} "
        f"windows_profitable={positive_windows}/{total_windows} "
        f"total_oos_trades={total_oos_trades} "
        f"avg_sortino={avg_sortino:.4f} "
        f"param_cv_max={param_cv_max:.4f} "
        f"wfo_efficiency={wfo_efficiency:.4f} "
        f"best_regime={best_regime}:{best_regime_pf:.2f} "
        f"worst_regime={worst_regime}:{worst_regime_pf:.2f} "
        f"loc={strategy_loc} loc_penalty={loc_penalty:.1f}"
    )

    # ── Save full results JSON ─────────────────────────────────────
    out_dir = "artifacts/autoresearch"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{strategy_name}_latest.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    emit(f"AUTORESEARCH_LOG {log_path}")

    # Restore stdout/stderr and close log
    sys.stdout = real_stdout
    sys.stderr = sys.__stderr__
    log_file.close()

    # Exit 0 = success
    sys.exit(0)


if __name__ == "__main__":
    main()
