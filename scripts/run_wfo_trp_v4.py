#!/usr/bin/env python
"""Robust Walk-Forward Optimization for Trend Rider Pro v4.

Uses full 2023-2025 data range (27 months):
  - Rolling 3-month train windows, 1-month test windows
  - 2-month holdout (Feb-Mar 2025) — never touched during optimization
  - 80 Optuna trials per window for good TPE convergence
  - ~18 WFO windows covering diverse market regimes

Run:
    cd /Users/jacobmcmillan/Empire/Cerberus
    uv run python scripts/run_wfo_trp_v4.py
"""

import json
import logging
import os
import sys
import time
import warnings

# Suppress all logging before any imports
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["EMPIRE_LOG_LEVEL"] = "CRITICAL"
os.environ.setdefault("EMPIRE_LOG_FORMAT", "json")

os.chdir("/Users/jacobmcmillan/Empire/Cerberus")
sys.path.insert(0, ".")

import yaml  # noqa: E402

from src.analytics.optuna_harness import WalkForwardOptimizer  # noqa: E402

STRATEGY = "trend_rider_pro"
DATA_DIR = "data/bars_2023_2025"
CONFIG_PATH = "config/backtest_v2.yaml"

# Full date range from our data
FULL_START = "2023-01-03"
FULL_END = "2025-03-19"

# WFO parameters
MIN_TRAIN_MONTHS = 3  # rolling 3-month training window
TEST_MONTHS = 1  # 1-month OOS test per window
HOLDOUT_MONTHS = 2  # Feb-Mar 2025 held out entirely
N_TRIALS = 80  # good TPE convergence with 7 params
MODE = "rolling"  # regime-adaptive, doesn't dilute with old data
WORKERS = 4  # parallel Optuna workers

with open(CONFIG_PATH) as f:
    base_config = yaml.safe_load(f)

# Ensure only TRP is enabled
for sname in base_config.get("strategies", {}):
    base_config["strategies"][sname]["enabled"] = sname == STRATEGY

wfo = WalkForwardOptimizer(
    full_start=FULL_START,
    full_end=FULL_END,
    min_train_months=MIN_TRAIN_MONTHS,
    test_months=TEST_MONTHS,
    n_trials=N_TRIALS,
    holdout_months=HOLDOUT_MONTHS,
    mode=MODE,
)

windows = wfo.get_windows()
holdout = wfo.get_holdout_window()

print(f"{'=' * 70}")
print("  TREND RIDER PRO v4 — Walk-Forward Optimization")
print(f"{'=' * 70}")
print(f"  Data range:  {FULL_START} → {FULL_END}")
print(f"  Mode:        {MODE} ({MIN_TRAIN_MONTHS}m train / {TEST_MONTHS}m test)")
print(f"  Holdout:     {holdout['start']} → {holdout['end']}")
print(f"  Windows:     {len(windows)}")
print(f"  Trials/win:  {N_TRIALS}")
print(f"  Workers:     {WORKERS}")
print(f"  Total trials: {len(windows) * N_TRIALS}")
print(f"{'=' * 70}")

for i, w in enumerate(windows):
    print(f"  Window {i + 1:2d}: Train {w['train_start']}→{w['train_end']}  Test {w['test_start']}→{w['test_end']}")
print(f"  Holdout:   {holdout['start']} → {holdout['end']} (UNTOUCHED)")
print(f"{'=' * 70}\n")

t0 = time.time()

results = wfo.run(
    strategy_name=STRATEGY,
    base_config=base_config,
    data_dir=DATA_DIR,
    config_path=CONFIG_PATH,
    workers=WORKERS,
)

elapsed = time.time() - t0

# Save results
out_path = f"artifacts/optimization/{STRATEGY}_wfo_v4_robust.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

# Summary
print(f"\n{'=' * 70}")
print(f"  RESULTS — {STRATEGY} WFO v4")
print(f"{'=' * 70}")
print(f"  WFO Efficiency: {results['wfo_efficiency_ratio']:.4f}  (target > 0.5)")
dsr = results.get("deflated_sharpe_ratio")
dsr_str = f"{dsr:.4f}" if dsr is not None else "N/A"
print(f"  Deflated Sharpe: {dsr_str}  (>0.95 = genuine)")
print(f"  IS scores:  {[f'{s:.3f}' for s in results['is_scores']]}")
print(f"  OOS scores: {[f'{s:.3f}' for s in results['oos_scores']]}")
print(f"  Total time: {elapsed / 60:.1f} minutes")
print("\n  Param stability:")
for param, stats in results.get("param_stability", {}).items():
    stable_str = "✓" if stats.get("stable") else "✗"
    print(f"    {param:25s} mean={stats['mean']:8.4f}  cv={stats['cv']:.4f}  {stable_str}")

if results.get("param_sensitivity"):
    print("\n  Param sensitivity (last window):")
    for ps in results["param_sensitivity"]:
        print(f"    {ps['param_name']:25s} corr={ps['correlation']:+.4f}  rank={ps['sensitivity_rank']}")

print(f"\n  Holdout window: {results['holdout_window']}")
print(f"  Results saved to: {out_path}")
print(f"{'=' * 70}")
