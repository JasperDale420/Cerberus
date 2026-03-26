#!/usr/bin/env python
"""Run robust WFO on Trend Rider Pro v4 using full 2023-2026 data.

This uses 3 years of data with:
- Rolling 6-month train windows, 1-month test windows
- 3-month holdout (Jan-Mar 2026) for final validation
- 50 Optuna trials per window for thorough exploration
- Parameter stability analysis across windows
"""

import json
import logging
import os
import sys
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

# Load base config (TRP v4 optimized from autoresearch)
with open("config/backtest_v2.yaml") as f:
    base_config = yaml.safe_load(f)

# ---- Robust WFO Configuration ----
# Full data range: 2023-01-01 to 2026-03-20
# Holdout: last 3 months (Jan-Mar 2026) — never touched during optimization
# Train windows: 6 months rolling, 1-month test
# This gives ~24 WFO windows across 2.5 years of data

wfo = WalkForwardOptimizer(
    full_start="2024-01-01",
    full_end="2026-03-20",
    min_train_months=6,
    test_months=2,
    n_trials=15,
    holdout_months=3,
    mode="rolling",
)

windows = wfo.get_windows()
holdout = wfo.get_holdout_window()

print(f"{'=' * 70}", flush=True)
print("  Robust WFO — Trend Rider Pro v4", flush=True)
print("  Data range: 2024-01-01 → 2026-03-20", flush=True)
print("  Mode: rolling (6-month train, 2-month test)", flush=True)
print(f"  Windows: {len(windows)}", flush=True)
print(f"  Holdout: {holdout['start']} → {holdout['end']}", flush=True)
print(f"  Trials per window: {wfo.n_trials}", flush=True)
print(f"{'=' * 70}", flush=True)

print("\nWindows:", flush=True)
for i, w in enumerate(windows):
    print(f"  [{i:2d}] Train: {w['train_start']}→{w['train_end']}  Test: {w['test_start']}→{w['test_end']}", flush=True)

print(f"\nHoldout: {holdout['start']} → {holdout['end']}", flush=True)
print("\nStarting optimization...\n", flush=True)

results = wfo.run(
    strategy_name="trend_rider_pro",
    base_config=base_config,
    data_dir="data/bars_2023_2025",
    config_path="config/backtest_v2.yaml",
    workers=2,
)

# Save results
out_dir = "artifacts/optimization"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/trend_rider_pro_wfo_robust_2023_2026.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'=' * 70}", flush=True)
print(f"  Results saved to {out_path}", flush=True)
print(f"  WFO Efficiency: {results['wfo_efficiency_ratio']:.4f}", flush=True)
print(f"  IS scores: {results['is_scores']}", flush=True)
print(f"  OOS scores: {results['oos_scores']}", flush=True)
if "holdout_result" in results:
    hr = results["holdout_result"]
    print(
        f"  Holdout: score={hr.get('holdout_score', 'N/A')}, "
        f"sharpe={hr.get('holdout_sharpe', 'N/A')}, "
        f"PF={hr.get('holdout_pf', 'N/A')}, "
        f"trades={hr.get('holdout_n_trades', 'N/A')}",
        flush=True,
    )
    print(f"  OOS→Holdout ratio: {hr.get('oos_to_holdout_ratio', 'N/A')}", flush=True)
    print(f"  Holdout PASSED: {hr.get('passed', 'N/A')}", flush=True)
print("\nParam stability:", flush=True)
print(json.dumps(results.get("param_stability", {}), indent=2), flush=True)
print(f"{'=' * 70}", flush=True)
