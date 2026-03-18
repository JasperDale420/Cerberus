#!/usr/bin/env python
"""Autoresearch verification script: run backtest and output a single composite score.

Usage:
    cd /Users/jacobmcmillan/Empire/Cerberus
    uv run python scripts/autoresearch_backtest.py

Outputs a JSON line with all metrics + composite_score to stdout.
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

from src.analytics.optuna_harness import composite_objective, run_backtest_for_optimization


def main():
    import yaml

    config_path = "config/backtest_v2.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Full year backtest
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    data_dir = "data/bars_2024"

    metrics = run_backtest_for_optimization(
        start_date=start_date,
        end_date=end_date,
        config=config,
        data_dir=data_dir,
        config_path=config_path,
    )

    # Compute composite score (same as Optuna objective)
    score = composite_objective(metrics, min_trades=10)

    # Also compute a PnL+Sharpe focused score for our goal
    net_pnl = metrics.get("net_pnl", 0.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    trade_sharpe = metrics.get("trade_sharpe_ratio", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    n_trades = metrics.get("n_trades", 0)

    # Our autoresearch metric: PnL-weighted composite
    # Higher = better
    autoresearch_score = (
        0.35 * (net_pnl / 1000.0)  # PnL in $K (dominant factor)
        + 0.25 * sharpe  # Daily Sharpe
        + 0.15 * trade_sharpe  # Trade-level Sharpe
        + 0.15 * pf  # Profit factor
        + 0.10 * min(n_trades / 100.0, 2.0)  # Trade count bonus (capped at 2.0)
    )

    output = {
        "autoresearch_score": round(autoresearch_score, 4),
        "composite_score": round(score, 4),
        "net_pnl": metrics.get("net_pnl", 0.0),
        "sharpe_ratio": sharpe,
        "trade_sharpe_ratio": trade_sharpe,
        "profit_factor": pf,
        "n_trades": n_trades,
        "winrate": metrics.get("winrate", 0.0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
        "avg_hold_minutes": metrics.get("avg_hold_minutes", 0.0),
        "avg_pnl": metrics.get("avg_pnl", 0.0),
        "calmar_ratio": metrics.get("calmar_ratio", 0.0),
        "total_return_pct": metrics.get("total_return_pct", 0.0),
        "final_equity": metrics.get("final_equity", 0.0),
    }

    # Print only the JSON result to stdout
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
