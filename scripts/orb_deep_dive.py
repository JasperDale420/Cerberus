#!/usr/bin/env python3
"""
Senior Quant ORB Strategy Deep Dive Analysis

Analyzes:
1. Trade-level distributions (wins/losses, holding periods)
2. Time-of-day within session
3. Symbol-level performance
4. Entry/Exit analysis (stop distance, target distance, R-multiples)
5. Gap characteristics (gap_pct in meta)
6. Opening range characteristics (OR width, breakout type)
7. VWAP relationship
8. Exit reasons
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")


def load_orb_trades():
    """Load ORB trades from low-slippage backtest."""
    path = RESULTS_DIR / "backtest_5yr_low_slip_orb.json"
    if not path.exists():
        print(f"Error: {path} not found")
        return []
    with open(path) as f:
        return json.load(f).get("engine_trades", [])


def parse_time(t_str):
    """Parse ISO datetime string."""
    if not t_str:
        return None
    try:
        return datetime.fromisoformat(t_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def analyze_win_loss_distribution(trades):
    """Analyze the distribution of wins and losses."""
    print("\n" + "=" * 70)
    print("  1. WIN/LOSS DISTRIBUTION")
    print("=" * 70)

    pnls = [t.get("pnl_net", 0) or 0 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    scratches = [p for p in pnls if p == 0]

    print(f"\n  Total trades: {len(trades):,}")
    print(f"  Winners: {len(wins):,} ({len(wins) / len(trades) * 100:.1f}%)")
    print(f"  Losers: {len(losses):,} ({len(losses) / len(trades) * 100:.1f}%)")
    print(f"  Scratches: {len(scratches):,}")

    if wins:
        print("\n  WINNING TRADES:")
        print(f"    Avg win: ${sum(wins) / len(wins):.2f}")
        print(f"    Median win: ${sorted(wins)[len(wins) // 2]:.2f}")
        print(f"    Max win: ${max(wins):.2f}")
        print(f"    Min win: ${min(wins):.2f}")

        # Win size buckets
        small_wins = [w for w in wins if w < 10]
        medium_wins = [w for w in wins if 10 <= w < 50]
        large_wins = [w for w in wins if w >= 50]
        print(f"    Small wins (<$10): {len(small_wins)} ({sum(small_wins):,.0f})")
        print(f"    Medium wins ($10-50): {len(medium_wins)} ({sum(medium_wins):,.0f})")
        print(f"    Large wins (>$50): {len(large_wins)} ({sum(large_wins):,.0f})")

    if losses:
        print("\n  LOSING TRADES:")
        print(f"    Avg loss: ${sum(losses) / len(losses):.2f}")
        print(f"    Median loss: ${sorted(losses)[len(losses) // 2]:.2f}")
        print(f"    Max loss: ${min(losses):.2f}")  # Most negative
        print(f"    Min loss: ${max(losses):.2f}")  # Least negative

        # Loss size buckets
        small_losses = [loss for loss in losses if loss > -10]
        medium_losses = [loss for loss in losses if -50 <= loss <= -10]
        large_losses = [loss for loss in losses if loss < -50]
        print(
            f"    Small losses (>-$10): {len(small_losses)} ({sum(small_losses):,.0f})"
        )
        print(
            f"    Medium losses (-$10 to -$50): {len(medium_losses)} ({sum(medium_losses):,.0f})"
        )
        print(
            f"    Large losses (<-$50): {len(large_losses)} ({sum(large_losses):,.0f})"
        )


def analyze_pnl_r_multiples(trades):
    """Analyze R-multiple performance."""
    print("\n" + "=" * 70)
    print("  2. R-MULTIPLE ANALYSIS (Risk-Adjusted Returns)")
    print("=" * 70)

    r_values = [t.get("pnl_r", 0) or 0 for t in trades if t.get("pnl_r") is not None]

    if not r_values:
        print("  No R-multiple data available")
        return

    positive_r = [r for r in r_values if r > 0]
    negative_r = [r for r in r_values if r < 0]

    print(f"\n  R-multiple stats (n={len(r_values):,}):")
    print(f"    Avg R: {sum(r_values) / len(r_values):.3f}")
    print(
        f"    Positive R trades: {len(positive_r)} ({len(positive_r) / len(r_values) * 100:.1f}%)"
    )
    print(
        f"    Negative R trades: {len(negative_r)} ({len(negative_r) / len(r_values) * 100:.1f}%)"
    )

    if positive_r:
        print(f"    Avg positive R: {sum(positive_r) / len(positive_r):.2f}")
    if negative_r:
        print(f"    Avg negative R: {sum(negative_r) / len(negative_r):.2f}")

    # R-bucket analysis
    buckets = defaultdict(lambda: {"count": 0, "pnl": 0})
    for t in trades:
        r = t.get("pnl_r", 0) or 0
        pnl = t.get("pnl_net", 0) or 0
        if r < -1:
            bucket = "< -1R"
        elif r < 0:
            bucket = "-1R to 0"
        elif r < 1:
            bucket = "0 to 1R"
        elif r < 2:
            bucket = "1R to 2R"
        else:
            bucket = "> 2R"
        buckets[bucket]["count"] += 1
        buckets[bucket]["pnl"] += pnl

    print("\n  R-Multiple Buckets:")
    for bucket in ["< -1R", "-1R to 0", "0 to 1R", "1R to 2R", "> 2R"]:
        if bucket in buckets:
            b = buckets[bucket]
            print(f"    {bucket:12}: {b['count']:5,} trades | PnL: ${b['pnl']:>10,.0f}")


def analyze_holding_period(trades):
    """Analyze holding period distribution."""
    print("\n" + "=" * 70)
    print("  3. HOLDING PERIOD ANALYSIS")
    print("=" * 70)

    holding_secs = [t.get("holding_period_seconds", 0) or 0 for t in trades]
    valid_holding = [h for h in holding_secs if h > 0]

    if not valid_holding:
        print("  No holding period data")
        return

    avg_mins = sum(valid_holding) / len(valid_holding) / 60
    median_idx = len(valid_holding) // 2
    median_mins = sorted(valid_holding)[median_idx] / 60

    print("\n  Holding Period:")
    print(f"    Avg: {avg_mins:.1f} minutes")
    print(f"    Median: {median_mins:.1f} minutes")
    print(f"    Min: {min(valid_holding) / 60:.1f} minutes")
    print(f"    Max: {max(valid_holding) / 60:.1f} minutes")

    # Bucket by holding time and calculate PnL
    buckets = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in trades:
        h = t.get("holding_period_seconds", 0) or 0
        pnl = t.get("pnl_net", 0) or 0

        if h < 300:  # < 5 min
            bucket = "< 5 min"
        elif h < 900:  # 5-15 min
            bucket = "5-15 min"
        elif h < 1800:  # 15-30 min
            bucket = "15-30 min"
        elif h < 3600:  # 30-60 min
            bucket = "30-60 min"
        else:
            bucket = "> 60 min"

        buckets[bucket]["count"] += 1
        buckets[bucket]["pnl"] += pnl
        if pnl > 0:
            buckets[bucket]["wins"] += 1

    print("\n  PnL by Holding Period:")
    for bucket in ["< 5 min", "5-15 min", "15-30 min", "30-60 min", "> 60 min"]:
        if bucket in buckets:
            b = buckets[bucket]
            wr = b["wins"] / b["count"] * 100 if b["count"] > 0 else 0
            avg_pnl = b["pnl"] / b["count"] if b["count"] > 0 else 0
            profitable = "✅" if b["pnl"] > 0 else "❌"
            print(
                f"    {profitable} {bucket:12}: {b['count']:5,} trades | WR={wr:5.1f}% | PnL: ${b['pnl']:>10,.0f} | Avg: ${avg_pnl:.2f}"
            )


def analyze_by_symbol(trades):
    """Analyze performance by symbol."""
    print("\n" + "=" * 70)
    print("  4. SYMBOL ANALYSIS")
    print("=" * 70)

    symbols = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in trades:
        sym = t.get("symbol", "unknown")
        pnl = t.get("pnl_net", 0) or 0
        symbols[sym]["count"] += 1
        symbols[sym]["pnl"] += pnl
        if pnl > 0:
            symbols[sym]["wins"] += 1

    # Sort by PnL
    sorted_symbols = sorted(symbols.items(), key=lambda x: x[1]["pnl"], reverse=True)

    print("\n  TOP 10 PROFITABLE SYMBOLS:")
    for sym, stats in sorted_symbols[:10]:
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        avg = stats["pnl"] / stats["count"] if stats["count"] > 0 else 0
        print(
            f"    ✅ {sym:6} | {stats['count']:4} trades | WR={wr:5.1f}% | PnL: ${stats['pnl']:>8,.0f} | Avg: ${avg:>6.2f}"
        )

    print("\n  BOTTOM 10 LOSING SYMBOLS:")
    for sym, stats in sorted_symbols[-10:]:
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        avg = stats["pnl"] / stats["count"] if stats["count"] > 0 else 0
        print(
            f"    ❌ {sym:6} | {stats['count']:4} trades | WR={wr:5.1f}% | PnL: ${stats['pnl']:>8,.0f} | Avg: ${avg:>6.2f}"
        )


def analyze_meta_characteristics(trades):
    """Analyze trade meta data - gap_pct, OR characteristics, etc."""
    print("\n" + "=" * 70)
    print("  5. ORB-SPECIFIC CHARACTERISTICS")
    print("=" * 70)

    # Breakout type analysis
    breakout_types = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    gap_buckets = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})

    for t in trades:
        meta = t.get("meta") or {}
        pnl = t.get("pnl_net", 0) or 0

        # Breakout type
        bt = meta.get("breakout_type", "unknown")
        breakout_types[bt]["count"] += 1
        breakout_types[bt]["pnl"] += pnl
        if pnl > 0:
            breakout_types[bt]["wins"] += 1

        # Gap size
        gap_pct = meta.get("gap_pct", 0) or 0
        if gap_pct < -0.02:
            gap_bucket = "Gap Down >2%"
        elif gap_pct < 0:
            gap_bucket = "Gap Down 0-2%"
        elif gap_pct < 0.02:
            gap_bucket = "Gap Up 0-2%"
        else:
            gap_bucket = "Gap Up >2%"

        gap_buckets[gap_bucket]["count"] += 1
        gap_buckets[gap_bucket]["pnl"] += pnl
        if pnl > 0:
            gap_buckets[gap_bucket]["wins"] += 1

    print("\n  By BREAKOUT TYPE:")
    for bt, stats in sorted(
        breakout_types.items(), key=lambda x: x[1]["pnl"], reverse=True
    ):
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        profitable = "✅" if stats["pnl"] > 0 else "❌"
        print(
            f"    {profitable} {bt:12} | {stats['count']:5,} trades | WR={wr:5.1f}% | PnL: ${stats['pnl']:>10,.0f}"
        )

    print("\n  By GAP SIZE:")
    for gap, stats in sorted(
        gap_buckets.items(), key=lambda x: x[1]["pnl"], reverse=True
    ):
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        profitable = "✅" if stats["pnl"] > 0 else "❌"
        print(
            f"    {profitable} {gap:16} | {stats['count']:5,} trades | WR={wr:5.1f}% | PnL: ${stats['pnl']:>10,.0f}"
        )


def analyze_entry_time_within_session(trades):
    """Analyze when during opening session trades are entered."""
    print("\n" + "=" * 70)
    print("  6. INTRA-SESSION TIMING")
    print("=" * 70)

    time_buckets = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})

    for t in trades:
        entry_time = parse_time(t.get("entry_time"))
        if not entry_time:
            continue

        # Get minute of day (in EST)
        hour = entry_time.hour
        minute = entry_time.minute
        total_mins = hour * 60 + minute

        # Market opens at 9:30 ET = 570 minutes
        mins_since_open = total_mins - (9 * 60 + 30)
        if mins_since_open < 0:
            continue  # Pre-market

        pnl = t.get("pnl_net", 0) or 0

        if mins_since_open < 15:
            bucket = "0-15 min"
        elif mins_since_open < 30:
            bucket = "15-30 min"
        elif mins_since_open < 45:
            bucket = "30-45 min"
        elif mins_since_open < 60:
            bucket = "45-60 min"
        else:
            bucket = "> 60 min"

        time_buckets[bucket]["count"] += 1
        time_buckets[bucket]["pnl"] += pnl
        if pnl > 0:
            time_buckets[bucket]["wins"] += 1

    print("\n  PnL by MINUTES SINCE MARKET OPEN:")
    for bucket in ["0-15 min", "15-30 min", "30-45 min", "45-60 min", "> 60 min"]:
        if bucket in time_buckets:
            b = time_buckets[bucket]
            if b["count"] > 0:
                wr = b["wins"] / b["count"] * 100
                avg = b["pnl"] / b["count"]
                profitable = "✅" if b["pnl"] > 0 else "❌"
                print(
                    f"    {profitable} {bucket:12}: {b['count']:5,} trades | WR={wr:5.1f}% | PnL: ${b['pnl']:>10,.0f} | Avg: ${avg:.2f}"
                )


def generate_optimization_recommendations(trades):
    """Generate specific actionable recommendations."""
    print("\n" + "=" * 70)
    print("  7. OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)

    # Symbol analysis
    symbols = defaultdict(lambda: {"count": 0, "pnl": 0})
    for t in trades:
        sym = t.get("symbol", "unknown")
        pnl = t.get("pnl_net", 0) or 0
        symbols[sym]["count"] += 1
        symbols[sym]["pnl"] += pnl

    profitable_symbols = [
        s for s, d in symbols.items() if d["pnl"] > 0 and d["count"] >= 100
    ]
    losing_symbols = [s for s, d in symbols.items() if d["pnl"] < -1000]

    # Holding period analysis
    short_hold = {"count": 0, "pnl": 0}
    long_hold = {"count": 0, "pnl": 0}
    for t in trades:
        h = t.get("holding_period_seconds", 0) or 0
        pnl = t.get("pnl_net", 0) or 0
        if h < 900:  # < 15 min
            short_hold["count"] += 1
            short_hold["pnl"] += pnl
        else:
            long_hold["count"] += 1
            long_hold["pnl"] += pnl

    print(f"""
  Based on the analysis, here are specific recommendations:

  1. SYMBOL FILTER:
     • Profitable symbols (>100 trades, positive PnL): {", ".join(profitable_symbols[:5])}
     • Consider EXCLUDING: {", ".join(losing_symbols[:5])}
     
  2. HOLDING PERIOD:
     • Short holds (<15 min): {short_hold["count"]:,} trades, PnL ${short_hold["pnl"]:,.0f}
     • Long holds (>15 min): {long_hold["count"]:,} trades, PnL ${long_hold["pnl"]:,.0f}
     • Recommendation: {"Keep positions shorter" if long_hold["pnl"] < short_hold["pnl"] else "Allow trades more time"}
     
  3. REGIME FILTER (from earlier analysis):
     • Only trade in: flat/down trends + high/low volatility
     • Avoid: up trends (highest losses)
     
  4. GAP FILTER:
     • Analyze whether small gaps or large gaps perform better
     • Consider min/max gap_pct thresholds
     
  5. EXECUTION:
     • Current slippage: 2.0 bps - strategy is profitable with good execution
     • Consider using limit orders instead of market orders at breakout
""")


def main():
    print("\n" + "=" * 70)
    print("  ORB STRATEGY DEEP DIVE - SENIOR QUANT ANALYSIS")
    print("  Data: 5-year backtest (2020-2024), Low Slippage (2 bps)")
    print("=" * 70)

    trades = load_orb_trades()
    if not trades:
        return

    total_pnl = sum(t.get("pnl_net", 0) or 0 for t in trades)
    wins = sum(1 for t in trades if (t.get("pnl_net", 0) or 0) > 0)

    print(
        f"\n  OVERVIEW: {len(trades):,} trades | WR={wins / len(trades) * 100:.1f}% | Net PnL=${total_pnl:,.0f}"
    )

    analyze_win_loss_distribution(trades)
    analyze_pnl_r_multiples(trades)
    analyze_holding_period(trades)
    analyze_by_symbol(trades)
    analyze_meta_characteristics(trades)
    analyze_entry_time_within_session(trades)
    generate_optimization_recommendations(trades)


if __name__ == "__main__":
    main()
