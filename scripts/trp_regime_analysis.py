"""TRP regime analysis — break down performance by month/quarter/symbol.

Runs backtest with iter10 best params, then analyzes per-trade data
to find periods and conditions where TRP is profitable vs not.
"""

from __future__ import annotations

import asyncio
import copy
import os
import sys
import time
from collections import defaultdict
from unittest.mock import patch as _mock_patch

import yaml

# Allow imports
sys.path.insert(0, os.getcwd())


def run_and_get_report():
    """Run backtest and return the raw BacktestReportCard (not .to_dict())."""
    from src.backtest.runner import run_backtest as _run_backtest_async
    from src.core.config import ConfigLoader

    config_path = "config/backtest_trp_sortino.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Trial DB
    trial_id = f"regime_{os.getpid()}_{int(time.time() * 1000) % 1_000_000}"
    db_dir = os.path.join(os.getcwd(), ".agents", "tmp", "regime_dbs")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{trial_id}.db")
    config["database_url"] = f"sqlite:///{db_path}"
    config["log_level"] = "ERROR"

    try:
        import src.core.logger as _logger_mod

        _logger_mod._configured = False
    except Exception:
        pass

    def _patched_load(self, config_path_or_dir=None):
        return copy.deepcopy(config)

    with _mock_patch.object(ConfigLoader, "load_config", _patched_load):
        result = asyncio.run(
            _run_backtest_async(
                start_date=config.get("start_date", "2024-01-01"),
                end_date=config.get("end_date", "2025-12-31"),
                config_path=config_path,
                data_dir="data/bars_2023_2025",
            )
        )

    # Clean up
    try:
        os.unlink(db_path)
    except Exception:
        pass

    return result


def analyze_trades(report):
    """Break down trades by month, quarter, symbol."""
    trades = report.trades

    if not trades:
        print("No trades found!")
        return

    print(f"\n{'=' * 80}")
    print(f"TREND RIDER PRO — REGIME ANALYSIS ({len(trades)} trades)")
    print(f"{'=' * 80}\n")

    # ── Per-Month Breakdown ──
    monthly = defaultdict(list)
    for t in trades:
        key = t.entry_time.strftime("%Y-%m")
        monthly[key].append(t.pnl)

    print("=" * 70)
    print(f"{'MONTH':<10} {'TRADES':>7} {'NET PnL':>12} {'AVG PnL':>10} {'WR%':>7} {'PF':>7}")
    print("-" * 70)
    for month in sorted(monthly.keys()):
        pnls = monthly[month]
        n = len(pnls)
        net = sum(pnls)
        avg = net / n if n else 0
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100 if n else 0
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p < 0))
        pf = gross_w / gross_l if gross_l > 0 else 999.0
        marker = " ✅" if net > 0 else ""
        print(f"{month:<10} {n:>7} {net:>12,.2f} {avg:>10,.2f} {wr:>6.1f}% {pf:>6.2f}{marker}")

    total_pnl = sum(t.pnl for t in trades)
    print("-" * 70)
    print(f"{'TOTAL':<10} {len(trades):>7} {total_pnl:>12,.2f}")

    # ── Per-Quarter Breakdown ──
    quarterly = defaultdict(list)
    for t in trades:
        q = (t.entry_time.month - 1) // 3 + 1
        key = f"{t.entry_time.year}-Q{q}"
        quarterly[key].append(t.pnl)

    print(f"\n{'=' * 70}")
    print(f"{'QUARTER':<10} {'TRADES':>7} {'NET PnL':>12} {'AVG PnL':>10} {'WR%':>7} {'PF':>7}")
    print("-" * 70)
    for quarter in sorted(quarterly.keys()):
        pnls = quarterly[quarter]
        n = len(pnls)
        net = sum(pnls)
        avg = net / n if n else 0
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100 if n else 0
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p < 0))
        pf = gross_w / gross_l if gross_l > 0 else 999.0
        marker = " ✅" if net > 0 else ""
        print(f"{quarter:<10} {n:>7} {net:>12,.2f} {avg:>10,.2f} {wr:>6.1f}% {pf:>6.2f}{marker}")

    # ── Per-Symbol Breakdown ──
    by_symbol = defaultdict(list)
    for t in trades:
        by_symbol[t.symbol].append(t.pnl)

    print(f"\n{'=' * 70}")
    print(f"{'SYMBOL':<10} {'TRADES':>7} {'NET PnL':>12} {'AVG PnL':>10} {'WR%':>7} {'PF':>7}")
    print("-" * 70)
    for sym in sorted(by_symbol.keys(), key=lambda s: sum(by_symbol[s]), reverse=True):
        pnls = by_symbol[sym]
        n = len(pnls)
        net = sum(pnls)
        avg = net / n if n else 0
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100 if n else 0
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p < 0))
        pf = gross_w / gross_l if gross_l > 0 else 999.0
        marker = " ✅" if net > 0 else ""
        print(f"{sym:<10} {n:>7} {net:>12,.2f} {avg:>10,.2f} {wr:>6.1f}% {pf:>6.2f}{marker}")

    # ── Day-of-Week Breakdown ──
    by_dow = defaultdict(list)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for t in trades:
        dow = t.entry_time.weekday()
        by_dow[dow].append(t.pnl)

    print(f"\n{'=' * 70}")
    print(f"{'DAY':<10} {'TRADES':>7} {'NET PnL':>12} {'AVG PnL':>10} {'WR%':>7} {'PF':>7}")
    print("-" * 70)
    for dow in range(5):
        if dow not in by_dow:
            continue
        pnls = by_dow[dow]
        n = len(pnls)
        net = sum(pnls)
        avg = net / n if n else 0
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100 if n else 0
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p < 0))
        pf = gross_w / gross_l if gross_l > 0 else 999.0
        marker = " ✅" if net > 0 else ""
        print(f"{days[dow]:<10} {n:>7} {net:>12,.2f} {avg:>10,.2f} {wr:>6.1f}% {pf:>6.2f}{marker}")

    # ── Entry Hour Breakdown ──
    by_hour = defaultdict(list)
    for t in trades:
        h = t.entry_time.hour
        by_hour[h].append(t.pnl)

    print(f"\n{'=' * 70}")
    print(f"{'HOUR':<10} {'TRADES':>7} {'NET PnL':>12} {'AVG PnL':>10} {'WR%':>7} {'PF':>7}")
    print("-" * 70)
    for h in sorted(by_hour.keys()):
        pnls = by_hour[h]
        n = len(pnls)
        net = sum(pnls)
        avg = net / n if n else 0
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100 if n else 0
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p < 0))
        pf = gross_w / gross_l if gross_l > 0 else 999.0
        marker = " ✅" if net > 0 else ""
        print(f"{h:02d}:00     {n:>7} {net:>12,.2f} {avg:>10,.2f} {wr:>6.1f}% {pf:>6.2f}{marker}")

    # ── Profitable vs losing periods summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY: Profitable Months")
    print("-" * 70)
    profitable_months = {m: pnls for m, pnls in monthly.items() if sum(pnls) > 0}
    losing_months = {m: pnls for m, pnls in monthly.items() if sum(pnls) <= 0}
    prof_pnl = sum(sum(pnls) for pnls in profitable_months.values())
    loss_pnl = sum(sum(pnls) for pnls in losing_months.values())
    print(
        f"  Profitable months: {len(profitable_months)}/{len(monthly)} ({len(profitable_months) / len(monthly) * 100:.0f}%)"
    )
    print(f"  Combined PnL from profitable months: ${prof_pnl:,.2f}")
    print(f"  Combined PnL from losing months: ${loss_pnl:,.2f}")
    print(f"  Net: ${prof_pnl + loss_pnl:,.2f}")


if __name__ == "__main__":
    print("Running backtest with iter10 best params...")
    report = run_and_get_report()
    if report is None:
        print("ERROR: Backtest returned None")
        sys.exit(1)
    if isinstance(report, dict):
        print("ERROR: Got dict instead of BacktestReportCard — no per-trade data available")
        print(f"Metrics: {report}")
        sys.exit(1)
    analyze_trades(report)
