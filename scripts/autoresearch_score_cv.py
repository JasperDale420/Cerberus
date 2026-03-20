"""Cross-validated autoresearch scoring across multiple regime windows.

Scores strategy performance across 5 non-overlapping 1-year windows.
Final score penalizes inconsistency: mean(scores) - 1.0 * std(scores).

Usage:
    cd /Users/jacobmcmillan/Empire/Cerberus
    uv run python scripts/autoresearch_score_cv.py
    uv run python scripts/autoresearch_score_cv.py --config config/backtest_portfolio.yaml
"""

import asyncio
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["EMPIRE_LOG_LEVEL"] = "CRITICAL"
os.environ.setdefault("EMPIRE_LOG_FORMAT", "json")

os.chdir("/Users/jacobmcmillan/Empire/Cerberus")
sys.path.insert(0, ".")

from src.backtest.runner import run_backtest  # noqa: E402

WINDOWS = [
    ("2020-06-01", "2021-06-01"),
    ("2021-06-01", "2022-06-01"),
    ("2022-06-01", "2023-06-01"),
    ("2023-06-01", "2024-06-01"),
    ("2024-06-01", "2025-06-01"),
]

CONFIG = "config/backtest_portfolio.yaml"
DATA_DIR = "data/bars_2023_2025"

MIN_PASSING_WINDOWS = 4


def compute_composite_score(metrics: dict) -> float:
    """Same scoring formula as autoresearch_score.py for consistency."""
    pnl = metrics.get("net_pnl", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    pf = metrics.get("profit_factor", 0)
    winrate = metrics.get("winrate", 0)
    n_trades = metrics.get("n_trades", 0)

    pnl_score = pnl / 10000.0
    sharpe_score = sharpe / 2.0
    pf_score = max(0, pf - 1.0)
    wr_score = (winrate - 0.50) / 0.15

    if n_trades < 50:
        trade_score = -1.0
    elif n_trades < 200:
        trade_score = (n_trades - 50) / 150.0 * 0.5
    elif n_trades <= 800:
        trade_score = 0.5 + 0.5 * (1.0 - abs(n_trades - 400) / 400.0)
    elif n_trades <= 2000:
        trade_score = max(0, 0.5 - (n_trades - 800) / 2400.0)
    else:
        trade_score = -0.5

    composite = 0.30 * pnl_score + 0.25 * sharpe_score + 0.20 * pf_score + 0.15 * wr_score + 0.10 * trade_score
    return round(composite, 4)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-validated autoresearch scoring")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    window_scores = []
    window_details = []

    for i, (start, end) in enumerate(WINDOWS):
        print(f"  Window {i + 1}/{len(WINDOWS)}: {start} → {end}", file=sys.stderr, flush=True)

        report = await run_backtest(start, end, args.config, data_dir=args.data_dir)
        if report is None:
            window_scores.append(-999)
            window_details.append({"window": f"{start}→{end}", "error": "backtest_failed"})
            continue

        metrics = report.to_dict()
        score = compute_composite_score(metrics)
        window_scores.append(score)
        window_details.append(
            {
                "window": f"{start}→{end}",
                "score": score,
                "net_pnl": round(metrics.get("net_pnl", 0), 2),
                "sharpe": round(metrics.get("sharpe_ratio", 0), 3),
                "profit_factor": round(metrics.get("profit_factor", 0), 2),
                "winrate": round(metrics.get("winrate", 0) * 100, 1),
                "n_trades": metrics.get("n_trades", 0),
                "max_drawdown_pct": round(metrics.get("max_drawdown_pct", 0), 2),
            }
        )

    # Cross-validated composite: penalize inconsistency
    valid_scores = [s for s in window_scores if s > -999]
    if len(valid_scores) < 3:
        cv_score = -999
        mean_score = -999
        std_score = 0
    else:
        mean_score = sum(valid_scores) / len(valid_scores)
        std_score = math.sqrt(sum((s - mean_score) ** 2 for s in valid_scores) / len(valid_scores))
        cv_score = round(mean_score - 1.0 * std_score, 4)
        mean_score = round(mean_score, 4)
        std_score = round(std_score, 4)

    # Viability gate: must pass in MIN_PASSING_WINDOWS windows
    passing_windows = sum(1 for s in valid_scores if s >= 0.0)
    passed_gate = passing_windows >= MIN_PASSING_WINDOWS

    result = {
        "autoresearch_score": cv_score,
        "mean_score": mean_score,
        "std_score": std_score,
        "passing_windows": passing_windows,
        "total_windows": len(WINDOWS),
        "viability_gate": passed_gate,
        "window_scores": window_scores,
        "window_details": window_details,
    }

    print(json.dumps(result), file=sys.stderr)
    print(f"AUTORESEARCH_SCORE={cv_score}")


if __name__ == "__main__":
    asyncio.run(main())
