#!/usr/bin/env python
"""Verification backtest with current config params."""

import asyncio
import json
import logging
import os
import sys
import warnings

os.chdir("/Users/jacobmcmillan/Empire/Cerberus")
sys.path.insert(0, ".")

# Suppress verbose logging
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")
os.environ["EMPIRE_LOG_LEVEL"] = "WARNING"
os.environ.setdefault("EMPIRE_LOG_FORMAT", "json")

from src.backtest.runner import run_backtest  # noqa: E402


async def main():
    print("Running verification backtest (2024-01-01 to 2024-12-31)...", flush=True)
    print("Config: config/backtest_v2.yaml", flush=True)
    print("Data: data/bars_2024", flush=True)

    result = await run_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        config_path="config/backtest_v2.yaml",
        data_dir="data/bars_2024",
    )

    # Convert to dict
    report_dict = result.to_dict()
    m = result.metrics

    # Save full results
    out_path = "results/verification_backtest.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 60}", flush=True)
    print("VERIFICATION BACKTEST RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Net PnL:        ${m.net_pnl:,.2f}", flush=True)
    print(f"  Total Return:   {m.total_return_pct:.2f}%", flush=True)
    print(f"  Sharpe (daily):  {m.sharpe_ratio:.3f}", flush=True)
    print(f"  Profit Factor:  {m.profit_factor:.2f}", flush=True)
    print(f"  Win Rate:       {m.winrate:.1%}", flush=True)
    print(f"  Total Trades:   {m.n_trades}", flush=True)
    print(f"  Max Drawdown:   {m.max_drawdown_pct:.2f}%", flush=True)
    print(f"  Avg Hold (min): {m.avg_hold_seconds / 60:.1f}", flush=True)

    if hasattr(result, "strategy_metrics") and result.strategy_metrics:
        print("\nPer-Strategy:", flush=True)
        for name, sm in result.strategy_metrics.items():
            print(
                f"  {name}: trades={sm.get('n_trades', 0)} PnL=${sm.get('net_pnl', 0):,.2f} "
                f"WR={sm.get('winrate', 0):.1%} PF={sm.get('profit_factor', 0):.2f}",
                flush=True,
            )

    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
