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

# WFO parameters — full available data window (2016-06-01 → 2026-03-19)
# Rolling 12-month train + 6-month OOS + 3-month final holdout.
# Coverage: 2016-17 low-vol bull, 2018 vol-spike, 2019-20 pre/covid crash, 2020-21 recovery,
# 2022 bear, 2023 recovery, 2024 bull, 2025-26 latest. ~18 OOS windows.
# Each iteration ~45-75 min depending on strategy complexity.
WFO_FULL_START = "2016-06-01"
WFO_FULL_END = "2026-03-19"
WFO_TRAIN_MONTHS = 12
WFO_TEST_MONTHS = 6
WFO_HOLDOUT_MONTHS = 3
WFO_MODE = "rolling"
DATA_DIR = "data/bars_2023_2025"
CONFIG_PATH = "config/backtest_v2.yaml"
STARTING_CAPITAL = 100_000.0  # for SPY benchmark comparison


def classify_window_regime(data_dir: str, start: str, end: str) -> str:
    """Classify the dominant trend/vol regime for an OOS window using optimized detector.

    Uses the grid-search-optimized 2-axis detector (81.9% accuracy):
    - Trend: Dual SMA crossover (fast=10, slow=40, flat_band=1%)
    - Vol: 30-day realized vol (LOW<8%, HIGH>20%, SHOCK>50%)

    If pre-labeled regime data exists in data/regime_labeled/, uses that directly.
    Otherwise falls back to computing from raw 1-minute bars.
    """
    # Try pre-labeled data first (fast path)
    # Honor CERBERUS_REGIME_DIR for v2 dataset (data/regime_labeled_v2/)
    regime_dir = Path(os.environ.get("CERBERUS_REGIME_DIR", "data/regime_labeled"))
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

    total_windows = len(oos_scores)
    total_oos_trades = sum(m.get("n_trades", 0) for m in oos_metrics)
    # "Profitable" now means actual positive net PnL on the window — NOT
    # "didn't trip a hard-reject sentinel". This is the honest measure.
    positive_windows = sum(1 for m in oos_metrics if m.get("net_pnl", 0.0) > 0)

    # Avg Sortino (only from windows with trades)
    sortinos = [m.get("sortino_ratio", 0) for m in oos_metrics if m.get("n_trades", 0) > 0]
    avg_sortino = sum(sortinos) / max(len(sortinos), 1)

    # Simplicity bonus/penalty based on strategy LOC (Karpathy: simpler is better).
    # L3 fix: count actual code statements via AST, not raw non-blank/non-comment
    # lines. Previous counter penalized strategies for docstrings — a 50-line
    # docstring + 30 lines of code reported as 80 LOC (real code = 30).
    strategy_path = Path(f"src/strategies/{strategy_name}.py")
    strategy_loc = 0
    if strategy_path.exists():
        try:
            import ast

            tree = ast.parse(strategy_path.read_text())
            # Count statements that are NOT bare expression-statements containing only
            # a string literal (i.e., docstrings). Walks the entire tree so docstrings
            # inside class/function bodies are also excluded.
            for node in ast.walk(tree):
                if isinstance(node, ast.stmt):
                    if (
                        isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                    ):
                        continue  # docstring or stray string-statement
                    strategy_loc += 1
        except Exception:
            # AST parse failed — fall back to old line-based counter
            lines = strategy_path.read_text().splitlines()
            strategy_loc = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))
    if strategy_loc > 100:
        loc_penalty = max(-2.0, -0.05 * (strategy_loc - 100))  # -0.05/line, capped at -2.0
    elif strategy_loc < 50 and strategy_loc > 0:
        loc_penalty = min(0.5, 0.02 * (50 - strategy_loc))  # +0.02/line, capped at +0.5
    else:
        loc_penalty = 0.0

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
    # When --target-regime is set, all subsequent metrics (windows, trades, PnL,
    # SPY benchmark) are restricted to OOS windows whose classified regime matches.
    # This prevents an UP+NORMAL specialist from being penalized for DOWN+HIGH
    # windows it intentionally avoids.
    target_regime = args.target_regime
    scoring_windows = list(range(total_windows))  # default: all windows
    regime_filter_no_match = False
    if target_regime:
        # M1 fix: previously `if target_regime and regime_stats.get(target_regime):`
        # silently fell open to all-windows when the filter matched zero windows.
        # FLAT+NORMAL phase (5% of data) was the most exposed: a regime specialist
        # got scored against all 18 windows, guaranteeing gate failure.
        # Now we always attempt the filter and signal explicitly when it matches none.
        scoring_windows = [
            i
            for i, w in enumerate(windows)
            if classify_window_regime(args.data_dir, w["test_start"], w["test_end"]) == target_regime
        ]
        target_trades = sum(oos_metrics[i].get("n_trades", 0) for i in scoring_windows)
        target_positive_pnl = sum(1 for i in scoring_windows if oos_metrics[i].get("net_pnl", 0.0) > 0)
        emit(
            f"REGIME_FILTERED target={target_regime} "
            f"windows={target_positive_pnl}/{len(scoring_windows)} "
            f"trades={target_trades}"
        )
        if not scoring_windows:
            emit(
                f"REGIME_FILTER_NOMATCH target={target_regime} — 0 of {len(windows)} windows classified as this regime"
            )
            regime_filter_no_match = True
        positive_windows = target_positive_pnl
        total_windows = len(scoring_windows)
        total_oos_trades = target_trades

    # ── SPY buy-and-hold benchmark + composite score ────────────────
    # Score is driven by ratio_vs_spy, gated on basic sanity checks. All
    # scoring windows count (hard-rejects included). Math: BOTH strategy and
    # SPY are compounded as growth factors so the comparison is apples-to-apples.
    GATE_FLOOR = -2.0  # visible negative; agent sees gate failure clearly
    MIN_WINDOWS_PROFITABLE_PCT = 40.0  # at least 40% of scoring windows must have net_pnl > 0
    MIN_TRADES_PER_WINDOW = 5  # avg lower bound

    # Compound the strategy's per-window returns (Bug C2 fix).
    # Previous code used sum(pnl)/$100k (additive) while SPY compounded — see
    # debug session 260430-1728 finding C2 for the reproduction. Identical
    # 5%/window strategy and SPY would report ratio=0.64 instead of ratio=1.0.
    strategy_total_growth = 1.0
    running_capital = float(STARTING_CAPITAL)
    for i in scoring_windows:
        window_pnl = float(oos_metrics[i].get("net_pnl", 0.0))
        if running_capital <= 0:
            # Capital wiped out — strategy total growth stays at whatever it is,
            # remaining windows have nothing to risk.
            break
        gf = 1.0 + (window_pnl / running_capital)
        gf = max(gf, 0.01)  # floor at -99% per window; pathological windows can't cause neg-growth artifacts
        strategy_total_growth *= gf
        running_capital *= gf
    strategy_return_pct = (strategy_total_growth - 1.0) * 100.0
    strategy_total_pnl = sum(oos_metrics[i].get("net_pnl", 0.0) for i in scoring_windows)

    spy_return_pct = float("nan")
    ratio_vs_spy = float("nan")
    benchmark_span = "n/a"
    benchmark_mode = "none"
    try:
        spy_bars_path = Path(args.data_dir) / "SPY_1Min.parquet"
        if spy_bars_path.exists() and scoring_windows:
            spy = pd.read_parquet(spy_bars_path, columns=["timestamp", "close"])
            spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True)
            # Compound per-window SPY returns so the benchmark covers exactly the
            # scoring windows (no gaps for filtered-out regimes).
            sw_starts = [pd.Timestamp(windows[i]["test_start"], tz="UTC") for i in scoring_windows]
            sw_ends = [pd.Timestamp(windows[i]["test_end"], tz="UTC") for i in scoring_windows]
            spy_total_return = 1.0
            n_segments = 0
            for i in scoring_windows:
                seg_start = pd.Timestamp(windows[i]["test_start"], tz="UTC")
                seg_end = pd.Timestamp(windows[i]["test_end"], tz="UTC")
                seg = spy[(spy["timestamp"] >= seg_start) & (spy["timestamp"] <= seg_end)]
                if len(seg) > 1:
                    spy_total_return *= float(seg.iloc[-1]["close"]) / float(seg.iloc[0]["close"])
                    n_segments += 1
            if n_segments > 0:
                spy_return_pct = (spy_total_return - 1.0) * 100.0
                benchmark_span = f"{n_segments} segments {min(sw_starts).date()}..{max(sw_ends).date()}"
                # Sign-safe ratio (Bug C1 fix). Three regimes:
                #   bull (SPY > +1%): standard ratio = strategy_return / spy_return — matches "2x SPY return" goal.
                #   bear (SPY < -1%): ratio = 1 + alpha/|spy| — alpha=0 -> 1.0 (matched SPY); alpha=|spy| -> 2.0 (broke even in bear).
                #   flat (|SPY| <= 1%): ratio = strategy_return_pct — absolute return as alpha proxy when benchmark is noise.
                if spy_return_pct > 1.0:
                    ratio_vs_spy = strategy_return_pct / spy_return_pct
                    benchmark_mode = "bull_ratio"
                elif spy_return_pct < -1.0:
                    alpha = strategy_return_pct - spy_return_pct
                    ratio_vs_spy = 1.0 + (alpha / abs(spy_return_pct))
                    benchmark_mode = "bear_alpha"
                else:
                    # Bug M6 fix: don't force gate floor when SPY is near-flat. Use absolute return as score.
                    # This makes FLAT-regime specialists evaluable.
                    ratio_vs_spy = strategy_return_pct
                    benchmark_mode = "flat_absolute"
    except Exception as e:
        emit(f"AUTORESEARCH_BENCHMARK_ERROR error={e}")

    # ── Apply gates ─────────────────────────────────────────────────
    windows_profitable_pct = (positive_windows / max(total_windows, 1)) * 100.0
    min_total_trades = total_windows * MIN_TRADES_PER_WINDOW
    gate_failures: list[str] = []
    if regime_filter_no_match:
        # M1 fix: when --target-regime matches zero classified windows, fail the
        # gate explicitly rather than silently scoring against all windows.
        gate_failures.append(f"regime_filter_no_match={target_regime}")
    if total_oos_trades < min_total_trades:
        gate_failures.append(f"trades={total_oos_trades}<{min_total_trades}")
    if windows_profitable_pct < MIN_WINDOWS_PROFITABLE_PCT:
        gate_failures.append(f"win_windows={windows_profitable_pct:.0f}%<{MIN_WINDOWS_PROFITABLE_PCT:.0f}%")
    if not (ratio_vs_spy == ratio_vs_spy):  # NaN check
        gate_failures.append("benchmark_unavailable")
    elif ratio_vs_spy <= 0:
        gate_failures.append(f"ratio_vs_spy={ratio_vs_spy:.2f}<=0")

    if gate_failures:
        composite_score = GATE_FLOOR
        score_explanation = "gates_failed:" + ",".join(gate_failures)
    else:
        # Honest composite = how many SPYs we beat. Goal: 2.0+
        # loc_penalty is multiplicative, scaled to keep semantic ("50% LOC bonus = 1.025x score,
        # 100% LOC penalty = 0.9x score"). loc_penalty range ±2 -> ±10% effective.
        loc_multiplier = 1.0 + (loc_penalty * 0.05)
        composite_score = float(ratio_vs_spy) * loc_multiplier
        # Param-stability penalty: unstable params (CV > 0.3) shrink the score
        if param_cv_max > 0.3:
            composite_score *= max(0.5, 1.0 - (param_cv_max - 0.3))
        score_explanation = f"ratio_vs_spy={ratio_vs_spy:.2f}*loc_mult={loc_multiplier:.3f}*cv_mult"

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
        f"loc={strategy_loc} loc_penalty={loc_penalty:.1f} "
        f"score_explanation={score_explanation}"
    )
    emit(
        f"AUTORESEARCH_BENCHMARK strategy_return_pct={strategy_return_pct:.2f} "
        f"strategy_total_pnl={strategy_total_pnl:.2f} "
        f"spy_return_pct={spy_return_pct:.2f} "
        f"ratio_vs_spy={ratio_vs_spy:.2f} "
        f"mode={benchmark_mode} "
        f"goal=2.00 "
        f"span={benchmark_span}"
    )

    # ── Save full results JSON ─────────────────────────────────────
    import subprocess

    try:
        commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit_sha = "unknown"
    results["meta"] = {
        "commit_sha": commit_sha,
        "strategy": strategy_name,
        "eval_completed": datetime.now().isoformat(),
        "wfo_window": f"{WFO_FULL_START}..{WFO_FULL_END}",
    }
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
