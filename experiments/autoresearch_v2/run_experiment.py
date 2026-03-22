#!/usr/bin/env python3
"""
Autoresearch V2 Experiment Runner

Runs backtests on IS (2020-2024) and OOS (2025) periods.
Extracts key metrics and prints machine-parseable summary.
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.getcwd())

from src.backtest.runner import run_backtest


async def run_period(config_path: str, start: str, end: str, data_dir: str, label: str):
    """Run a single backtest period and return metrics dict."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Running {label}: {start} → {end}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    report = await run_backtest(
        start_date=start,
        end_date=end,
        config_path=config_path,
        data_dir=data_dir,
    )

    if report is None:
        return {
            "label": label,
            "total_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "benchmark_return_pct": 0.0,
            "alpha_pct": 0.0,
            "calmar": 0.0,
            "sortino": 0.0,
        }

    m = report.metrics
    bm = m.benchmark if m.benchmark else None
    bm_ret = bm.benchmark_return * 100.0 if bm and hasattr(bm, "benchmark_return") and bm.benchmark_return else 0.0

    return {
        "label": label,
        "total_return_pct": round(m.total_return_pct, 2),
        "sharpe": round(m.sharpe_ratio, 3),
        "max_drawdown_pct": round(m.max_drawdown_pct, 2),
        "total_trades": m.n_trades or 0,
        "win_rate": round(m.winrate * 100, 1) if m.winrate else 0.0,
        "profit_factor": round(m.profit_factor, 2) if m.profit_factor else 0.0,
        "benchmark_return_pct": round(bm_ret, 2),
        "alpha_pct": round(m.total_return_pct - bm_ret, 2),
        "calmar": round(m.calmar_ratio, 3) if m.calmar_ratio else 0.0,
        "sortino": round(m.sortino_ratio, 3) if m.sortino_ratio else 0.0,
        "cagr": round(m.cagr, 2) if m.cagr else 0.0,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="data/bars_2023_2025")
    parser.add_argument("--is-start", default="2020-01-02")
    parser.add_argument("--is-end", default="2024-12-31")
    parser.add_argument("--oos-start", default="2025-01-02")
    parser.add_argument("--oos-end", default="2025-12-31")
    parser.add_argument("--oos-only", action="store_true")
    parser.add_argument("--is-only", action="store_true")
    args = parser.parse_args()

    results = {}

    if not args.oos_only:
        results["in_sample"] = await run_period(
            args.config, args.is_start, args.is_end, args.data_dir, "IN-SAMPLE (2020-2024)"
        )

    if not args.is_only:
        results["out_of_sample"] = await run_period(
            args.config, args.oos_start, args.oos_end, args.data_dir, "OUT-OF-SAMPLE (2025)"
        )

    # Print summary
    print("\n" + "=" * 60)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for _key, r in results.items():
        print(f"\n--- {r['label']} ---")
        print(f"  Total Return:     {r['total_return_pct']:+.2f}%")
        print(f"  Benchmark (SPY):  {r['benchmark_return_pct']:+.2f}%")
        print(f"  Alpha:            {r['alpha_pct']:+.2f}%")
        print(f"  Sharpe:           {r['sharpe']:.3f}")
        print(f"  Sortino:          {r['sortino']:.3f}")
        print(f"  Max Drawdown:     {r['max_drawdown_pct']:.2f}%")
        print(f"  Calmar:           {r['calmar']:.3f}")
        print(f"  Trades:           {r['total_trades']}")
        print(f"  Win Rate:         {r['win_rate']:.1f}%")
        print(f"  Profit Factor:    {r['profit_factor']:.2f}")

    print(f"\nJSON_RESULTS: {json.dumps(results)}")
    return results


if __name__ == "__main__":
    asyncio.run(main())
