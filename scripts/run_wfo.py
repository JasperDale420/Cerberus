#!/usr/bin/env python
"""Run WFO for a single strategy."""

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

strategy_name = sys.argv[1]

with open("config/backtest_v2.yaml") as f:
    base_config = yaml.safe_load(f)

wfo = WalkForwardOptimizer(
    full_start="2024-01-01",
    full_end="2024-12-31",
    min_train_months=3,
    test_months=1,
    n_trials=25,
    holdout_months=2,
    mode="rolling",
)

print(f"Windows: {wfo.get_windows()}", flush=True)
print(f"Holdout: {wfo.get_holdout_window()}", flush=True)

results = wfo.run(
    strategy_name=strategy_name,
    base_config=base_config,
    data_dir="data/bars_2024",
    config_path="config/backtest_v2.yaml",
    workers=2,
)

# Save results to a known location
out_path = f"artifacts/optimization/{strategy_name}_wfo_post_fix.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {out_path}", flush=True)
print(f"WFO Efficiency: {results['wfo_efficiency_ratio']}", flush=True)
print(f"IS scores: {results['is_scores']}", flush=True)
print(f"OOS scores: {results['oos_scores']}", flush=True)
print(f"Param stability: {json.dumps(results['param_stability'], indent=2)}", flush=True)
