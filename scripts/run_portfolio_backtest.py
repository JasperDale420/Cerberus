#!/usr/bin/env python
"""Run multi-strategy portfolio backtest across multiple regime windows.

Evaluates the portfolio across 6 non-overlapping windows spanning 2020-2026
to measure diversification benefit and per-strategy contribution.

Usage:
    cd /Users/jacobmcmillan/Empire/Cerberus
    uv run python scripts/run_portfolio_backtest.py
    uv run python scripts/run_portfolio_backtest.py --window 3  # run only window 3
"""

import asyncio
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["EMPIRE_LOG_LEVEL"] = "WARNING"
os.environ.setdefault("EMPIRE_LOG_FORMAT", "json")

os.chdir("/Users/jacobmcmillan/Empire/Cerberus")
sys.path.insert(0, ".")

from src.backtest.runner import run_backtest  # noqa: E402

WINDOWS = [
    ("2020-06-01", "2021-06-01", "COVID recovery → bull"),
    ("2021-06-01", "2022-06-01", "Peak → bear market"),
    ("2022-06-01", "2023-06-01", "Bear → recovery"),
    ("2023-06-01", "2024-06-01", "AI bull"),
    ("2024-06-01", "2025-06-01", "Mixed/choppy"),
    ("2025-06-01", "2026-03-20", "Recent"),
]

CONFIG = "config/backtest_portfolio.yaml"
DATA_DIR = "data/bars_2023_2025"


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-strategy portfolio backtest")
    parser.add_argument("--window", type=int, default=None, help="Run only window N (0-indexed)")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    windows = WINDOWS
    if args.window is not None:
        if 0 <= args.window < len(WINDOWS):
            windows = [WINDOWS[args.window]]
        else:
            print(f"ERROR: --window must be 0-{len(WINDOWS) - 1}")
            sys.exit(1)

    results = {}
    for start, end, label in windows:
        print(f"\n{'=' * 60}")
        print(f"  Window: {start} → {end} ({label})")
        print(f"{'=' * 60}", flush=True)

        report = await run_backtest(start, end, args.config, data_dir=args.data_dir)
        if report is None:
            print("  FAILED — no report")
            results[label] = {"error": "backtest_failed"}
            continue

        metrics = report.to_dict()
        results[label] = {
            "start": start,
            "end": end,
            "net_pnl": round(metrics.get("net_pnl", 0), 2),
            "sharpe": round(metrics.get("sharpe_ratio", 0), 3),
            "profit_factor": round(metrics.get("profit_factor", 0), 2),
            "winrate": round(metrics.get("winrate", 0) * 100, 1),
            "n_trades": metrics.get("n_trades", 0),
            "max_drawdown_pct": round(metrics.get("max_drawdown_pct", 0), 2),
            "calmar": round(metrics.get("calmar_ratio", 0), 3),
        }
        print(
            f"  PnL: ${metrics.get('net_pnl', 0):,.2f}  "
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f}  "
            f"Trades: {metrics.get('n_trades', 0)}",
            flush=True,
        )

    out_dir = "artifacts/portfolio"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/portfolio_baseline_2020_2026.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  Results saved to {out_path}")
    print(f"{'=' * 60}")
    for label, m in results.items():
        if "error" not in m:
            print(f"  {label:30s}  PnL=${m['net_pnl']:>10,.2f}  Sharpe={m['sharpe']:>6.3f}  Trades={m['n_trades']:>4d}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
