#!/usr/bin/env python
"""Autoresearch scoring: Trend Rider Pro Sortino + PnL composite.

Uses run_backtest_for_optimization for fast iteration cycles.

Usage:
    cd /Users/jacobmcmillan/Empire/Cerberus
    uv run python scripts/autoresearch_trp_sortino.py
    uv run python scripts/autoresearch_trp_sortino.py --start 2026-01-01 --end 2026-03-19  # OOS

Output:
    AUTORESEARCH_SCORE=<float>   (higher is better)
"""

import json
import logging
import math
import os
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["EMPIRE_LOG_LEVEL"] = "CRITICAL"
os.environ.setdefault("EMPIRE_LOG_FORMAT", "json")

os.chdir("/Users/jacobmcmillan/Empire/Cerberus")
sys.path.insert(0, ".")

from src.analytics.optuna_harness import run_backtest_for_optimization  # noqa: E402

IS_START = "2024-01-01"
IS_END = "2025-12-31"
CONFIG_PATH = "config/backtest_trp_sortino.yaml"
DATA_DIR = "data/bars_2023_2025"


def compute_sortino_pnl_score(metrics: dict) -> float:
    """Composite score emphasizing Sortino ratio and Net PnL.

    Weights:
        40% blended Sortino  +  35% PnL  +  15% Profit Factor  +  10% trade bonus
        - Drawdown penalty (exponential above 15%)
        - Friction penalty for sub-10m holds

    Hard-rejects:
        - < 20 trades
        - Negative expectancy (avg_pnl <= 0)
        - Negative Sortino
    """
    n_trades = metrics.get("n_trades", 0)
    if n_trades < 20:
        return -1000.0

    avg_pnl = metrics.get("avg_pnl", 0.0)
    if avg_pnl <= 0:
        return -500.0

    # Blended Sortino: trade-level for short holds, daily for long holds
    daily_sortino = metrics.get("sortino_ratio", 0.0)
    trade_sortino = metrics.get("trade_sortino_ratio", 0.0)
    avg_hold_min = metrics.get("avg_hold_minutes", 999)

    if not math.isclose(trade_sortino, 0.0, abs_tol=1e-9):
        if avg_hold_min < 60:
            sortino = 0.7 * trade_sortino + 0.3 * daily_sortino
        elif avg_hold_min > 120:
            sortino = 0.3 * trade_sortino + 0.7 * daily_sortino
        else:
            sortino = 0.5 * trade_sortino + 0.5 * daily_sortino
    else:
        sortino = daily_sortino

    if sortino < 0:
        return -300.0

    sortino = max(-5.0, min(sortino, 15.0))

    net_pnl = metrics.get("net_pnl", 0.0)
    pnl_score = net_pnl / 10000.0

    pf = metrics.get("profit_factor", 0.0)
    pf = max(0.0, min(pf, 10.0))
    pf_score = max(0.0, pf - 1.0)

    max_dd = metrics.get("max_drawdown_pct", 100.0)

    dd_penalty = 0.0
    if max_dd > 15.0:
        dd_penalty = (max_dd - 15.0) ** 1.5

    trade_bonus = min(n_trades / 200.0, 1.0)

    friction_penalty = 0.0
    if avg_hold_min < 5:
        friction_penalty = 2.0
    elif avg_hold_min < 10:
        friction_penalty = 1.0
    elif avg_hold_min < 20:
        friction_penalty = 0.3

    score = (
        0.40 * sortino + 0.35 * pnl_score + 0.15 * pf_score + 0.10 * trade_bonus - 0.05 * dd_penalty - friction_penalty
    )
    return round(score, 4)


def main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=IS_START)
    parser.add_argument("--end", default=IS_END)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    metrics = run_backtest_for_optimization(
        start_date=args.start,
        end_date=args.end,
        config=config,
        data_dir=args.data_dir,
        config_path=args.config,
    )

    score = compute_sortino_pnl_score(metrics)

    result = {
        "autoresearch_score": score,
        "net_pnl": round(metrics.get("net_pnl", 0), 2),
        "sortino_daily": round(metrics.get("sortino_ratio", 0), 3),
        "sortino_trade": round(metrics.get("trade_sortino_ratio", 0), 3),
        "sharpe": round(metrics.get("sharpe_ratio", 0), 3),
        "profit_factor": round(metrics.get("profit_factor", 0), 2),
        "winrate": round(metrics.get("winrate", 0) * 100, 1),
        "n_trades": metrics.get("n_trades", 0),
        "max_drawdown_pct": round(metrics.get("max_drawdown_pct", 0), 2),
        "total_return_pct": round(metrics.get("total_return_pct", 0), 2),
        "calmar": round(metrics.get("calmar_ratio", 0), 3),
        "avg_pnl": round(metrics.get("avg_pnl", 0), 2),
        "avg_hold_minutes": round(metrics.get("avg_hold_minutes", 0), 1),
        "max_consec_losers": metrics.get("max_consecutive_losers", 0),
    }

    print(json.dumps(result), file=sys.stderr)
    print(f"AUTORESEARCH_SCORE={score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
