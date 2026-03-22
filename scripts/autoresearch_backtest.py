#!/usr/bin/env python
"""Autoresearch 2025 verification: run backtest and output total_return_pct vs SPY.

Usage:
    cd /Users/jacobmcmillan/Empire/Cerberus
    uv run python scripts/autoresearch_backtest.py

Outputs a JSON line with total_return_pct and SPY comparison.
The METRIC line is total_return_pct — higher is better, goal is to beat SPY.
Exit code 0 = success, 1 = failure.
"""

import json
import logging
import os
import sys
import warnings

# Suppress noisy logs
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["EMPIRE_LOG_LEVEL"] = "CRITICAL"
os.environ.setdefault("EMPIRE_LOG_FORMAT", "json")

os.chdir("/Users/jacobmcmillan/Empire/Cerberus")
sys.path.insert(0, ".")

from src.analytics.optuna_harness import run_backtest_for_optimization  # noqa: E402


def main():
    import yaml

    config_path = "config/autoresearch_2025.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 2025 full year backtest
    start_date = "2025-01-02"
    end_date = "2025-12-31"
    data_dir = "data/bars_2023_2025"

    metrics = run_backtest_for_optimization(
        start_date=start_date,
        end_date=end_date,
        config=config,
        data_dir=data_dir,
        config_path=config_path,
    )

    total_return = metrics.get("total_return_pct", 0.0)
    benchmark_return = 0.0
    if "benchmark" in metrics and metrics["benchmark"]:
        bm = metrics["benchmark"]
        if isinstance(bm, dict):
            benchmark_return = bm.get("benchmark_return_pct", 0.0)
        elif hasattr(bm, "benchmark_return_pct"):
            benchmark_return = bm.benchmark_return_pct

    alpha = total_return - benchmark_return

    output = {
        "total_return_pct": round(total_return, 2),
        "benchmark_return_pct": round(benchmark_return, 2),
        "alpha": round(alpha, 2),
        "n_trades": metrics.get("n_trades", 0),
        "winrate": round(metrics.get("winrate", 0.0), 4),
        "profit_factor": round(metrics.get("profit_factor", 0.0), 4),
        "sharpe_ratio": round(metrics.get("sharpe_ratio", 0.0), 4),
        "max_drawdown_pct": round(metrics.get("max_drawdown_pct", 0.0), 2),
        "final_equity": round(metrics.get("final_equity", 0.0), 2),
        "net_pnl": round(metrics.get("net_pnl", 0.0), 2),
    }

    # Print JSON result
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
