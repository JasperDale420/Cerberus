#!/usr/bin/env python
"""Run WFO for all viable strategies with a reduced universe for speed."""

import json
import logging
import os
import sys
import time
import warnings

import yaml

os.chdir("/Users/jacobmcmillan/Empire/Cerberus")
sys.path.insert(0, ".")

# Suppress ALL logging to maximize speed — we only care about print() output
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["EMPIRE_LOG_LEVEL"] = "CRITICAL"
os.environ.setdefault("EMPIRE_LOG_FORMAT", "json")

from src.analytics.optuna_harness import WalkForwardOptimizer  # noqa: E402

# Minimal universe for fast optimization — 4 liquid symbols
OPT_SYMBOLS = ["SPY", "QQQ", "NVDA", "TSLA"]

STRATEGIES = [
    "rsi_bounce",
    "momentum_fade",
    # trend_rider_pro — already optimized (WFO v3: efficiency 1.21, DSR 1.0)
    # orb_v2 — already optimized (WFO v3: efficiency 2.17, DSR 1.0)
    # mean_reversion_pro excluded — generates <5 trades per window, not viable for WFO
    # flow_alpha excluded — needs live options flow data
]

with open("config/backtest_v2.yaml") as f:
    base_config = yaml.safe_load(f)

# Override universe for speed
base_config["universe"] = {"symbols": OPT_SYMBOLS}
# Use in-memory DB to avoid file I/O overhead
base_config["database_url"] = "sqlite://"
# Suppress logging from each trial
base_config["log_level"] = "CRITICAL"

wfo = WalkForwardOptimizer(
    full_start="2024-01-01",
    full_end="2024-12-31",
    min_train_months=3,
    test_months=1,
    n_trials=20,
    holdout_months=2,
    mode="rolling",
)

print(f"Windows: {len(wfo.get_windows())}", flush=True)
print(f"Holdout: {wfo.get_holdout_window()}", flush=True)
print(f"Symbols: {OPT_SYMBOLS}", flush=True)
print("Trials per window: 20", flush=True)
print("Workers: 2", flush=True)
print(f"Strategies: {STRATEGIES}", flush=True)
print(f"{'=' * 60}", flush=True)

all_results = {}

for i, strategy_name in enumerate(STRATEGIES):
    t0 = time.time()
    print(f"\n{'#' * 60}", flush=True)
    print(f"# Strategy {i + 1}/{len(STRATEGIES)}: {strategy_name}", flush=True)
    print(f"{'#' * 60}", flush=True)

    try:
        results = wfo.run(
            strategy_name=strategy_name,
            base_config=base_config,
            data_dir="data/bars_2024",
            config_path="config/backtest_v2.yaml",
            workers=2,
        )

        # Save individual results
        out_path = f"artifacts/optimization/{strategy_name}_wfo_v4.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        elapsed = time.time() - t0
        print(f"\n>>> {strategy_name} DONE in {elapsed / 60:.1f} min", flush=True)
        print(f"    Efficiency: {results['wfo_efficiency_ratio']}", flush=True)
        print(f"    IS scores:  {results['is_scores']}", flush=True)
        print(f"    OOS scores: {results['oos_scores']}", flush=True)
        all_results[strategy_name] = results

    except Exception as e:
        import traceback

        elapsed = time.time() - t0
        print(f"\n>>> {strategy_name} FAILED after {elapsed / 60:.1f} min: {e}", flush=True)
        traceback.print_exc()
        all_results[strategy_name] = {"error": str(e)}

# Summary
print(f"\n{'=' * 60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'=' * 60}", flush=True)
for name, res in all_results.items():
    if "error" in res:
        print(f"  {name}: ERROR — {res['error']}")
    else:
        eff = res.get("wfo_efficiency_ratio", 0)
        oos = res.get("oos_scores", [])
        valid_oos = [s for s in oos if s > -100]
        oos_avg = sum(valid_oos) / max(len(valid_oos), 1)
        print(f"  {name}: efficiency={eff:.3f}  OOS_avg={oos_avg:.3f}")

# Save combined summary
with open("artifacts/optimization/wfo_all_summary.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("\nAll results saved to artifacts/optimization/wfo_all_summary.json", flush=True)
