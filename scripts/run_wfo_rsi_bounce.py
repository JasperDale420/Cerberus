#!/usr/bin/env python
"""WFO for RSI Bounce — autoresearch hardening loop.

Expanded run using bars_2023_2025 dataset:
- 6-month train windows, 2-month test windows (more trades per OOS window)
- Full 2021-2025 range covers bull, bear, and recovery regimes
- 3-month holdout (Jan-Mar 2026) for final validation
- 30 Optuna trials per window for better param search
- 20-symbol universe for generalization across sectors
- Parameter stability analysis across windows

Exit code 0 = success. Prints JSON metrics to stdout for parsing.
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

# Load base config
with open("config/backtest_v2.yaml") as f:
    base_config = yaml.safe_load(f)

# 20-symbol universe: mega-cap tech + financials + energy + consumer + ETFs
# Covers multiple sectors for true generalization
OPT_SYMBOLS = [
    # ETFs (market / tech)
    "SPY",
    "QQQ",
    # Mega-cap tech
    "AAPL",
    "NVDA",
    "TSLA",
    "AMD",
    "AMZN",
    "META",
    "MSFT",
    "GOOGL",
    # Financials
    "JPM",
    "GS",
    "BAC",
    # Energy
    "XOM",
    "CVX",
    # Consumer / other
    "NFLX",
    "UBER",
    "COIN",
    # High-vol / meme-adjacent (stress test)
    "PLTR",
    "SOFI",
]

base_config["universe"] = {"symbols": OPT_SYMBOLS}
base_config["database_url"] = "sqlite://"
base_config["log_level"] = "CRITICAL"

# Enable rsi_bounce for this run
base_config["strategies"]["rsi_bounce"]["enabled"] = True

# Disable all other strategies
for s in base_config.get("strategies", {}):
    if s != "rsi_bounce":
        base_config["strategies"][s]["enabled"] = False

wfo = WalkForwardOptimizer(
    full_start="2021-01-01",
    full_end="2025-12-31",
    min_train_months=6,
    test_months=2,
    n_trials=30,
    holdout_months=3,
    mode="rolling",
)

windows = wfo.get_windows()
holdout = wfo.get_holdout_window()

print(f"{'=' * 60}", flush=True)
print("  WFO — RSI Bounce Hardening (EXPANDED)", flush=True)
print(f"  Windows: {len(windows)}", flush=True)
print(f"  Holdout: {holdout['start']} → {holdout['end']}", flush=True)
print(f"  Symbols: {len(OPT_SYMBOLS)} ({OPT_SYMBOLS})", flush=True)
print(f"  Trials per window: {wfo.n_trials}", flush=True)
print("  Data: bars_2023_2025 (2020-01 → 2026-03)", flush=True)
print(f"{'=' * 60}", flush=True)

for i, w in enumerate(windows):
    print(f"  [{i:2d}] Train: {w['train_start']}→{w['train_end']}  Test: {w['test_start']}→{w['test_end']}", flush=True)

print("\nStarting optimization...\n", flush=True)

results = wfo.run(
    strategy_name="rsi_bounce",
    base_config=base_config,
    data_dir="data/bars_2023_2025",
    config_path="config/backtest_v2.yaml",
    workers=2,
)

# Save results
out_dir = "artifacts/optimization"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/rsi_bounce_wfo_hardening.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

# Print summary metrics for autoresearch parsing
oos_scores = results.get("oos_scores", [])
oos_metrics = results.get("oos_metrics", [])

positive_windows = sum(1 for s in oos_scores if s > 0)
worst_sortino = min(
    (m.get("sortino_ratio", -999) for m in oos_metrics if m.get("n_trades", 0) > 0),
    default=-999,
)
avg_sortino = sum(m.get("sortino_ratio", 0) for m in oos_metrics if m.get("n_trades", 0) > 0) / max(
    sum(1 for m in oos_metrics if m.get("n_trades", 0) > 0), 1
)
total_oos_trades = sum(m.get("n_trades", 0) for m in oos_metrics)
avg_winrate = sum(m.get("winrate", 0) for m in oos_metrics if m.get("n_trades", 0) > 0) / max(
    sum(1 for m in oos_metrics if m.get("n_trades", 0) > 0), 1
)

print(f"\n{'=' * 60}", flush=True)
print("  RESULTS SUMMARY", flush=True)
print(f"  OOS scores: {oos_scores}", flush=True)
print(f"  Positive windows: {positive_windows}/{len(oos_scores)}", flush=True)
print(f"  Worst OOS Sortino: {worst_sortino:.3f}", flush=True)
print(f"  Avg OOS Sortino: {avg_sortino:.3f}", flush=True)
print(f"  Total OOS trades: {total_oos_trades}", flush=True)
print(f"  Avg OOS winrate: {avg_winrate:.2%}", flush=True)
print(f"  WFO Efficiency: {results.get('wfo_efficiency_ratio', 'N/A')}", flush=True)
print(f"  Results saved to {out_path}", flush=True)
print(f"{'=' * 60}", flush=True)

# Param stability
print("\nParam stability:", flush=True)
stability = results.get("param_stability", {})
for param, stats in sorted(stability.items()):
    flag = "✓" if stats.get("stable", False) else "✗ UNSTABLE"
    print(f"  {param:30s} mean={stats['mean']:.4f}  CV={stats['cv']:.4f}  {flag}", flush=True)

# Print parseable metric line for autoresearch
print(
    f"\nMETRIC: positive_windows={positive_windows} worst_sortino={worst_sortino:.3f} avg_sortino={avg_sortino:.3f}",
    flush=True,
)
