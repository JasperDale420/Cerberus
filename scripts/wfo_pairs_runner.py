"""Walk-Forward Optimization runner for Pair Trading V2.

Runs rolling WFO with 12-month train / 3-month test windows across
2020-01 to 2026-03, with pair-trading-appropriate thresholds.

Key adjustments vs standard WFO:
  - Enables pair_trading_v2 in ALL regimes (not just chop)
  - Lower min_trades thresholds (pairs naturally trade less)
  - Disables all other strategies
  - Suppresses verbose logging

Usage:
    uv run python scripts/wfo_pairs_runner.py
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import yaml

# Suppress verbose logging before any imports that configure loggers
logging.getLogger().setLevel(logging.WARNING)
for handler in logging.getLogger().handlers:
    handler.setLevel(logging.WARNING)

# Disable optuna's default INFO logging
import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.analytics.optuna_harness import WalkForwardOptimizer  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CONFIG_PATH = "config/backtest_v2.yaml"
DATA_DIR = "data/bars_2023_2025"
FULL_START = "2020-01-02"
FULL_END = "2026-03-15"

# WFO parameters
MIN_TRAIN_MONTHS = 12
TEST_MONTHS = 3
HOLDOUT_MONTHS = 6  # Last 6 months reserved for holdout validation (Sep 2025 → Mar 2026)
N_TRIALS = 25  # Trials per window (balance thoroughness vs runtime)
MODE = "rolling"  # Rolling windows for regime adaptation
STRATEGY = "pair_trading_v2"


def load_and_prepare_config() -> dict:
    """Load config and modify for pair-trading-only optimization."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    modified = copy.deepcopy(config)

    # 1. Disable all strategies except pair_trading_v2
    for strat_name, strat_cfg in modified.get("strategies", {}).items():
        if strat_name != STRATEGY:
            strat_cfg["enabled"] = False

    # 2. Enable pair_trading_v2 in ALL regime routing (not just chop)
    #    This is critical — the default config only routes it in 'chop',
    #    which severely limits trade opportunities during optimization.
    for regime in modified.get("strategy_routing", {}):
        routing = modified["strategy_routing"][regime]
        if STRATEGY not in routing:
            routing.append(STRATEGY)

    # 3. Suppress logging to reduce backtest overhead
    modified["log_level"] = "ERROR"

    # 4. Daily bar resolution — pair trading evaluates on daily close
    #    This aggregates 1-min bars to one per trading day, cutting bar count
    #    from ~978k to ~1289 per symbol and making WFO 2500x faster.
    modified.setdefault("backtest", {})["bar_resolution_minutes"] = "daily"

    return modified


def main() -> None:
    t0 = time.time()

    print(f"{'=' * 60}")
    print("Pair Trading V2 — Walk-Forward Optimization")
    print(f"{'=' * 60}")
    print(f"  Period:     {FULL_START} → {FULL_END}")
    print(f"  Train:      {MIN_TRAIN_MONTHS} months (rolling)")
    print(f"  Test:       {TEST_MONTHS} months")
    print(f"  Holdout:    {HOLDOUT_MONTHS} months")
    print(f"  Trials:     {N_TRIALS} per window")
    print(f"  Data:       {DATA_DIR}")
    print(f"{'=' * 60}")

    config = load_and_prepare_config()

    wfo = WalkForwardOptimizer(
        full_start=FULL_START,
        full_end=FULL_END,
        min_train_months=MIN_TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        n_trials=N_TRIALS,
        holdout_months=HOLDOUT_MONTHS,
        mode=MODE,
    )

    # Preview windows
    windows = wfo.get_windows()
    holdout = wfo.get_holdout_window()
    print(f"\n  WFO Windows: {len(windows)}")
    for i, w in enumerate(windows):
        print(f"    [{i + 1}] Train {w['train_start']}→{w['train_end']}  Test {w['test_start']}→{w['test_end']}")
    print(f"    Holdout: {holdout['start']} → {holdout['end']}")
    print(flush=True)

    # Run WFO
    result = wfo.run(
        strategy_name=STRATEGY,
        base_config=config,
        data_dir=DATA_DIR,
        config_path=CONFIG_PATH,
        workers=4,  # Parallel workers for faster optimization
        min_trades_per_month=0.5,  # IS: 34 pairs → ~0.5 trade/pair/month floor
        oos_min_trades_per_month=0,  # OOS: allow 0-trade windows → neutral score (0.0)
    )

    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"WFO Complete — {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")
    print(f"  Strategy: {result['strategy']}")
    print(f"  Windows:  {len(result['windows'])}")
    print(f"  WFO Efficiency: {result.get('wfo_efficiency_ratio', 0):.4f}")

    if result.get("best_params_per_window"):
        print("\n  Best Params per Window:")
        for i, params in enumerate(result["best_params_per_window"]):
            print(f"    [{i + 1}] {params}")

    if result.get("param_stability"):
        print("\n  Parameter Stability (CV < 0.30 = stable):")
        for param, stats in result["param_stability"].items():
            stable_str = "✓" if stats.get("stable") else "✗"
            print(f"    {stable_str} {param}: mean={stats['mean']:.4f} std={stats['std']:.4f} cv={stats['cv']:.4f}")

    print(f"\n  IS Scores:  {result.get('is_scores', [])}")
    print(f"  OOS Scores: {result.get('oos_scores', [])}")

    # Save full results to JSON
    output_path = Path("artifacts/optimization/runs") / STRATEGY / "wfo_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    # Print holdout results if available
    if result.get("holdout_result"):
        hr = result["holdout_result"]
        print("\n  Holdout Validation:")
        print(f"    Score:   {hr.get('holdout_score', 'N/A')}")
        print(f"    Sharpe:  {hr.get('holdout_sharpe', 'N/A')}")
        print(f"    PF:      {hr.get('holdout_pf', 'N/A')}")
        print(f"    Trades:  {hr.get('holdout_n_trades', 'N/A')}")
        print(f"    Passed:  {hr.get('passed', 'N/A')}")


if __name__ == "__main__":
    main()
