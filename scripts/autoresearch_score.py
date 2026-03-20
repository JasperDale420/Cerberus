"""Autoresearch scoring wrapper for Trend Rider Pro backtests.

Runs the backtest, extracts metrics, and computes a composite score.
Outputs a single JSON line with all metrics for the autoresearch loop to parse.

Usage:
    uv run python scripts/autoresearch_score.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

import asyncio
import json
import sys

from src.backtest.runner import run_backtest

START = "2025-01-02"
END = "2025-12-31"
CONFIG = "config/backtest_v2.yaml"
DATA_DIR = "data/bars_2023_2025"


def compute_composite_score(metrics: dict) -> float:
    """Compute weighted composite score from backtest metrics.

    Components (weights sum to 1.0):
      30% - Net PnL (normalized: $0 = 0, $10k = 1.0, uncapped)
      25% - Sharpe ratio (daily, normalized: 0 = 0, 2.0 = 1.0)
      20% - Profit factor (normalized: 1.0 = 0, 2.0 = 1.0)
      15% - Win rate (normalized: 50% = 0, 65% = 1.0)
      10% - Trade count penalty (too few or too many penalized)
    """
    pnl = metrics.get("net_pnl", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    pf = metrics.get("profit_factor", 0)
    winrate = metrics.get("winrate", 0)
    n_trades = metrics.get("n_trades", 0)

    # PnL component: $10k = 1.0
    pnl_score = pnl / 10000.0

    # Sharpe component: 2.0 = 1.0
    sharpe_score = sharpe / 2.0

    # Profit factor component: PF 2.0 = 1.0, PF 1.0 = 0
    pf_score = max(0, pf - 1.0)

    # Win rate component: 65% = 1.0, 50% = 0
    wr_score = (winrate - 0.50) / 0.15

    # Trade count: sweet spot 200-800 trades for a year
    if n_trades < 50:
        trade_score = -1.0  # Too few trades — not enough signal
    elif n_trades < 200:
        trade_score = (n_trades - 50) / 150.0 * 0.5
    elif n_trades <= 800:
        trade_score = 0.5 + 0.5 * (1.0 - abs(n_trades - 400) / 400.0)
    elif n_trades <= 2000:
        trade_score = max(0, 0.5 - (n_trades - 800) / 2400.0)
    else:
        trade_score = -0.5  # Over-trading

    composite = 0.30 * pnl_score + 0.25 * sharpe_score + 0.20 * pf_score + 0.15 * wr_score + 0.10 * trade_score

    return round(composite, 4)


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args()

    report = await run_backtest(args.start, args.end, CONFIG, data_dir=DATA_DIR)

    if report is None:
        result = {"autoresearch_score": -999, "error": "backtest_failed"}
        print(json.dumps(result))
        sys.exit(1)

    metrics = report.to_dict()
    score = compute_composite_score(metrics)

    result = {
        "autoresearch_score": score,
        "net_pnl": round(metrics.get("net_pnl", 0), 2),
        "sharpe": round(metrics.get("sharpe_ratio", 0), 3),
        "profit_factor": round(metrics.get("profit_factor", 0), 2),
        "winrate": round(metrics.get("winrate", 0) * 100, 1),
        "n_trades": metrics.get("n_trades", 0),
        "max_drawdown_pct": round(metrics.get("max_drawdown_pct", 0), 2),
        "total_return_pct": round(metrics.get("total_return_pct", 0), 2),
        "calmar": round(metrics.get("calmar_ratio", 0), 3),
        "avg_pnl": round(metrics.get("avg_pnl", 0), 2),
        "max_consec_losers": metrics.get("max_consecutive_losers", 0),
    }

    # Print to stderr for logging, stdout for machine parsing
    print(json.dumps(result), file=sys.stderr)
    print(f"AUTORESEARCH_SCORE={score}")


if __name__ == "__main__":
    asyncio.run(main())
