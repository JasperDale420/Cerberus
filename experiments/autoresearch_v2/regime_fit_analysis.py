#!/usr/bin/env python3
"""
Regime-Fit Analysis: Empirically determine which regimes each strategy performs best in.

1. Compute daily regime classification from SPY data (ADX-based + vol + trend)
2. Run each strategy solo via backtest
3. Parse trade log from report, join with regime by entry date
4. Output per-strategy, per-regime performance matrix
"""

import asyncio
import json
import os
import sys
from collections import defaultdict, deque
from datetime import date, datetime

import pandas as pd

sys.path.insert(0, os.getcwd())


# ── Regime Classification (matches regime_adaptive_momentum's internal logic) ──


def compute_daily_regimes(spy_path: str, start: str = "2020-01-02", end: str = "2025-12-31") -> dict[date, dict]:
    """Compute daily regime for each trading day from SPY 1-min bars."""
    df = pd.read_parquet(spy_path)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")

    # Filter to date range
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC")
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

    # Filter RTH
    et_times = df["timestamp"].dt.tz_convert("America/New_York")
    rth = (et_times.dt.time >= pd.Timestamp("09:30").time()) & (et_times.dt.time < pd.Timestamp("16:00").time())
    df = df[rth]

    # Aggregate to daily OHLCV
    daily = df.groupby(df["timestamp"].dt.date).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    daily = daily.sort_index()

    closes = deque(maxlen=60)
    highs = deque(maxlen=60)
    lows = deque(maxlen=60)

    regimes: dict[date, dict] = {}

    for dt_date, row in daily.iterrows():
        closes.append(row["close"])
        highs.append(row["high"])
        lows.append(row["low"])

        if len(closes) < 30:
            regimes[dt_date] = {"regime": "WARMUP", "adx": 0, "trend": "UNKNOWN", "vol_state": "UNKNOWN"}
            continue

        # EMA 20/50
        ema20 = _ema(list(closes), 20)
        ema50 = _ema(list(closes), 50) if len(closes) >= 50 else ema20

        # ADX
        adx = _adx(list(highs), list(lows), list(closes), 14)

        # ATR and avg ATR
        atr = _atr(list(highs), list(lows), list(closes), 14)
        atr_vals = []
        h, lo, c = list(highs), list(lows), list(closes)
        for i in range(max(1, len(c) - 20), len(c)):
            if i >= 1:
                tr = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
                atr_vals.append(tr)
        avg_atr = sum(atr_vals) / len(atr_vals) if atr_vals else (atr or 1.0)

        # Trend direction
        price = closes[-1]
        if ema20 and ema50:
            if ema20 > ema50 and price > ema20:
                trend = "UP"
            elif ema20 < ema50 and price < ema20:
                trend = "DOWN"
            else:
                trend = "FLAT"
        else:
            trend = "FLAT"

        # Volatility state
        if atr and avg_atr > 0 and atr > avg_atr * 2.0:
            vol_state = "SHOCK"
        elif atr and avg_atr > 0 and atr > avg_atr * 1.5:
            vol_state = "HIGH"
        elif atr and avg_atr > 0 and atr < avg_atr * 0.7:
            vol_state = "LOW"
        else:
            vol_state = "NORMAL"

        # Regime classification
        if vol_state == "SHOCK":
            regime = "SHOCK"
        elif adx is not None and adx < 20:
            regime = "RANGE_BOUND"
        elif adx is not None and adx >= 25:
            regime = "TRENDING"
        else:
            regime = "WEAK_TREND"

        regimes[dt_date] = {
            "regime": regime,
            "adx": round(adx or 0, 1),
            "trend": trend,
            "vol_state": vol_state,
        }

    return regimes


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    mult = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * mult + ema * (1 - mult)
    return ema


def _atr(highs: list, lows: list, closes: list, period: int = 14) -> float | None:
    if len(highs) < period + 1:
        return None
    trs = []
    for i in range(1, min(period + 1, len(highs))):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None


def _adx(highs: list, lows: list, closes: list, period: int = 14) -> float | None:
    n = len(highs)
    if n < 2 * period + 1:
        return None
    plus_dm, minus_dm, tr_vals = [], [], []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        tr_vals.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    if len(tr_vals) < 2 * period:
        return None
    s_plus = sum(plus_dm[:period])
    s_minus = sum(minus_dm[:period])
    s_tr = sum(tr_vals[:period])
    dx_vals = []
    for i in range(period, len(tr_vals)):
        s_plus = s_plus - (s_plus / period) + plus_dm[i]
        s_minus = s_minus - (s_minus / period) + minus_dm[i]
        s_tr = s_tr - (s_tr / period) + tr_vals[i]
        if s_tr > 0:
            pdi = 100.0 * s_plus / s_tr
            mdi = 100.0 * s_minus / s_tr
        else:
            pdi = mdi = 0.0
        di_sum = pdi + mdi
        dx = 100.0 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0.0
        dx_vals.append(dx)
    if len(dx_vals) < period:
        return None
    adx = sum(dx_vals[:period]) / period
    for i in range(period, len(dx_vals)):
        adx = (adx * (period - 1) + dx_vals[i]) / period
    return adx


# ── Backtest Runner ──


async def run_solo_strategy(strategy_name: str, config_template: dict, data_dir: str, start: str, end: str):
    """Run a single strategy in isolation and return the report."""
    import copy
    import tempfile

    import yaml

    from src.backtest.runner import run_backtest

    config = copy.deepcopy(config_template)

    # Disable all strategies except the target
    for name in list(config.get("strategies", {}).keys()):
        if name != strategy_name:
            config["strategies"][name]["enabled"] = False
        else:
            config["strategies"][name]["enabled"] = True

    # Write temp config
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix=f"solo_{strategy_name}_")
    os.close(fd)
    with open(path, "w") as f:
        yaml.dump(config, f)

    try:
        report = await run_backtest(start_date=start, end_date=end, config_path=path, data_dir=data_dir)
        return report
    finally:
        os.unlink(path)


# ── Main Analysis ──


async def main():
    import yaml

    data_dir = "data/bars_2023_2025"
    config_path = "experiments/autoresearch_v2/config_multi.yaml"

    with open(config_path) as f:
        config_template = yaml.safe_load(f)

    strategies = [name for name, cfg in config_template.get("strategies", {}).items() if cfg.get("enabled", True)]

    print("=" * 70)
    print("  REGIME-FIT ANALYSIS")
    print("=" * 70)

    # Step 1: Compute daily regimes
    print("\n[1/3] Computing daily regime classifications from SPY data...")
    spy_path = os.path.join(data_dir, "SPY_1Min.parquet")
    regimes = compute_daily_regimes(spy_path, "2020-01-02", "2025-12-31")

    regime_counts = defaultdict(int)
    for r in regimes.values():
        regime_counts[r["regime"]] += 1
    print(f"  Regime distribution ({sum(regime_counts.values())} trading days):")
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / sum(regime_counts.values()) * 100
        print(f"    {regime:20s}: {count:4d} days ({pct:.1f}%)")

    # Step 2: Run each strategy solo
    print(f"\n[2/3] Running {len(strategies)} strategies solo on 2020-2025...")
    all_results: dict[str, list[dict]] = {}

    for strat_name in strategies:
        print(f"\n  Running {strat_name}...")
        report = await run_solo_strategy(strat_name, config_template, data_dir, "2020-01-02", "2025-12-31")

        if report is None or not report.trades:
            print(f"    No trades for {strat_name}")
            all_results[strat_name] = []
            continue

        # Tag each trade with regime at entry date
        tagged_trades = []
        for t in report.trades:
            entry_date = t.entry_time.date() if isinstance(t.entry_time, datetime) else t.entry_time
            regime_info = regimes.get(
                entry_date, {"regime": "UNKNOWN", "trend": "UNKNOWN", "vol_state": "UNKNOWN", "adx": 0}
            )

            tagged_trades.append(
                {
                    "symbol": t.symbol,
                    "pnl": t.pnl,
                    "entry_date": str(entry_date),
                    "regime": regime_info["regime"],
                    "trend": regime_info["trend"],
                    "vol_state": regime_info["vol_state"],
                    "adx": regime_info["adx"],
                    "won": t.pnl > 0,
                }
            )

        all_results[strat_name] = tagged_trades
        print(f"    {len(tagged_trades)} trades tagged with regime data")

    # Step 3: Analyze per-regime performance
    print("\n" + "=" * 70)
    print("  PER-STRATEGY, PER-REGIME PERFORMANCE")
    print("=" * 70)

    regime_labels = ["TRENDING", "WEAK_TREND", "RANGE_BOUND", "SHOCK", "WARMUP"]

    for strat_name in strategies:
        trades = all_results.get(strat_name, [])
        if not trades:
            print(f"\n--- {strat_name}: NO TRADES ---")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {strat_name} ({len(trades)} total trades)")
        print(f"{'=' * 60}")
        print(f"  {'Regime':<15} {'Trades':>7} {'WinRate':>8} {'NetPnL':>10} {'AvgPnL':>9} {'PF':>6}")
        print(f"  {'-' * 55}")

        for regime in regime_labels:
            regime_trades = [t for t in trades if t["regime"] == regime]
            if not regime_trades:
                continue
            n = len(regime_trades)
            wins = sum(1 for t in regime_trades if t["won"])
            wr = wins / n * 100 if n > 0 else 0
            net_pnl = sum(t["pnl"] for t in regime_trades)
            avg_pnl = net_pnl / n if n > 0 else 0
            gross_profit = sum(t["pnl"] for t in regime_trades if t["pnl"] > 0)
            gross_loss = abs(sum(t["pnl"] for t in regime_trades if t["pnl"] < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

            marker = " ✓" if net_pnl > 0 and n >= 5 else " ✗" if net_pnl < 0 and n >= 5 else ""
            print(f"  {regime:<15} {n:>7} {wr:>7.1f}% {net_pnl:>+10.0f} {avg_pnl:>+9.1f} {pf:>6.2f}{marker}")

        # Also by trend direction
        print("\n  By TREND direction:")
        print(f"  {'Trend':<15} {'Trades':>7} {'WinRate':>8} {'NetPnL':>10} {'AvgPnL':>9}")
        print(f"  {'-' * 50}")
        for trend in ["UP", "DOWN", "FLAT"]:
            trend_trades = [t for t in trades if t["trend"] == trend]
            if not trend_trades:
                continue
            n = len(trend_trades)
            wins = sum(1 for t in trend_trades if t["won"])
            wr = wins / n * 100
            net_pnl = sum(t["pnl"] for t in trend_trades)
            avg_pnl = net_pnl / n
            print(f"  {trend:<15} {n:>7} {wr:>7.1f}% {net_pnl:>+10.0f} {avg_pnl:>+9.1f}")

        # By vol state
        print("\n  By VOLATILITY:")
        print(f"  {'VolState':<15} {'Trades':>7} {'WinRate':>8} {'NetPnL':>10} {'AvgPnL':>9}")
        print(f"  {'-' * 50}")
        for vol in ["LOW", "NORMAL", "HIGH", "SHOCK"]:
            vol_trades = [t for t in trades if t["vol_state"] == vol]
            if not vol_trades:
                continue
            n = len(vol_trades)
            wins = sum(1 for t in vol_trades if t["won"])
            wr = wins / n * 100
            net_pnl = sum(t["pnl"] for t in vol_trades)
            avg_pnl = net_pnl / n
            print(f"  {vol:<15} {n:>7} {wr:>7.1f}% {net_pnl:>+10.0f} {avg_pnl:>+9.1f}")

    # Save raw data for further analysis
    output_path = "experiments/autoresearch_v2/regime_fit_data.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw data saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
