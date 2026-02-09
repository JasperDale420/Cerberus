#!/usr/bin/env python3
"""
Senior Quant Analysis of Backtest Results
- Yearly performance breakdown
- Win/Loss asymmetry analysis
- Expectancy calculation
- Transaction cost impact
- Small-sample profitable conditions (looking for edge)
- Statistical significance
"""

import json
import math
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")
STRATEGIES = [
    "orb",
    "vwap_reversion",
    "trend_pullback",
    "vwap_trend_rider",
    "failed_breakout",
    "index_mean_reversion",
    "gap_fill",
]


def load_trades(strategy: str):
    path = RESULTS_DIR / f"backtest_5yr_{strategy}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("engine_trades", [])


def parse_year(entry_time):
    """Extract year from entry_time string."""
    if isinstance(entry_time, str):
        return int(entry_time[:4])
    return 2020


def analyze_strategy_deep(strategy: str, trades):
    """Deep quant analysis of a strategy."""
    if not trades:
        return

    print(f"\n{'=' * 80}")
    print(f"  {strategy.upper()} - DEEP QUANT ANALYSIS")
    print(f"{'=' * 80}")

    # Basic stats
    pnls = [t.get("pnl_net", 0) or 0 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Expectancy = (WR * avg_win) - ((1-WR) * abs(avg_loss))
    expectancy = (win_rate * avg_win) + (
        (1 - win_rate) * avg_loss
    )  # avg_loss is negative

    # Profit factor = gross profit / gross loss
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    print("\n  CORE METRICS:")
    print(f"    Trades: {len(trades):,}")
    print(f"    Win Rate: {win_rate * 100:.1f}%")
    print(f"    Total PnL: ${total_pnl:,.0f}")
    print(f"    Avg Win: ${avg_win:,.2f}")
    print(f"    Avg Loss: ${avg_loss:,.2f}")
    print(f"    Win/Loss Ratio: {abs(avg_win / avg_loss) if avg_loss else 0:.2f}")
    print(f"    Expectancy per trade: ${expectancy:,.2f}")
    print(f"    Profit Factor: {profit_factor:.3f}")

    # Slippage/commission impact
    total_slippage = sum(t.get("slippage_estimate", 0) or 0 for t in trades)
    total_commission = sum(t.get("commission", 0) or 0 for t in trades)
    pnl_gross = sum(t.get("pnl_gross", 0) or 0 for t in trades)

    print("\n  TRANSACTION COST ANALYSIS:")
    print(f"    Gross PnL: ${pnl_gross:,.0f}")
    print(f"    Slippage: ${total_slippage:,.0f}")
    print(f"    Commission: ${total_commission:,.0f}")
    print(f"    Net PnL: ${total_pnl:,.0f}")
    print(
        f"    Cost Impact: ${pnl_gross - total_pnl:,.0f} ({((pnl_gross - total_pnl) / abs(total_pnl) * 100) if total_pnl else 0:.1f}% of loss)"
    )

    # Yearly breakdown
    yearly = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
    for t in trades:
        year = parse_year(t.get("entry_time", "2020"))
        pnl = t.get("pnl_net", 0) or 0
        yearly[year]["trades"] += 1
        yearly[year]["pnl"] += pnl
        if pnl > 0:
            yearly[year]["wins"] += 1

    print("\n  YEARLY BREAKDOWN (looking for in-sample vs out-of-sample decay):")
    for year in sorted(yearly.keys()):
        y = yearly[year]
        wr = y["wins"] / y["trades"] if y["trades"] else 0
        profitable = "✅" if y["pnl"] > 0 else "❌"
        print(
            f"    {profitable} {year}: {y['trades']:,} trades | WR={wr * 100:.1f}% | PnL=${y['pnl']:,.0f}"
        )

    # Look for ANY profitable micro-conditions (min 20 trades for rough signal)
    print("\n  PROFITABLE MICRO-CONDITIONS (>= 20 trades):")
    combos = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
    for t in trades:
        regime = t.get("regime_tags_at_entry", {})
        key = (
            regime.get("session"),
            regime.get("trend"),
            regime.get("vol"),
            regime.get("risk"),
        )
        pnl = t.get("pnl_net", 0) or 0
        combos[key]["trades"] += 1
        combos[key]["pnl"] += pnl
        if pnl > 0:
            combos[key]["wins"] += 1

    profitable_combos = [
        (k, v) for k, v in combos.items() if v["pnl"] > 0 and v["trades"] >= 20
    ]
    profitable_combos.sort(key=lambda x: x[1]["pnl"], reverse=True)

    if profitable_combos:
        for key, stats in profitable_combos[:5]:
            wr = stats["wins"] / stats["trades"]
            session, trend, vol, risk = key
            print(f"    ✅ session={session} | trend={trend} | vol={vol} | risk={risk}")
            print(
                f"       {stats['trades']} trades | WR={wr * 100:.1f}% | PnL=${stats['pnl']:,.0f}"
            )
    else:
        print("    None found - strategy has no edge in any regime combination")

    # Statistical significance - is the negative result random?
    # Use simple t-test approximation
    if len(pnls) > 1:
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_dev = math.sqrt(variance)
        std_error = std_dev / math.sqrt(len(pnls))
        t_stat = mean_pnl / std_error if std_error > 0 else 0

        print("\n  STATISTICAL SIGNIFICANCE:")
        print(f"    Mean PnL per trade: ${mean_pnl:,.2f}")
        print(f"    Std Dev: ${std_dev:,.2f}")
        print(f"    t-statistic: {t_stat:.2f}")
        if abs(t_stat) > 2:
            print("    Result: STATISTICALLY SIGNIFICANT (t > 2)")
        else:
            print("    Result: NOT statistically significant (noise)")


def main():
    print("\n" + "=" * 80)
    print("  SENIOR QUANT ANALYSIS - 5-YEAR BACKTEST DATA")
    print("=" * 80)

    for strategy in STRATEGIES:
        trades = load_trades(strategy)
        if trades:
            analyze_strategy_deep(strategy, trades)

    # Summary recommendations
    print("\n" + "=" * 80)
    print("  QUANT RESEARCHER SUMMARY")
    print("=" * 80)
    print("""
  KEY FINDINGS:
  
  1. EDGE ANALYSIS: No strategy shows positive expectancy across any meaningful
     sample size. This suggests either:
     a) The strategies fundamentally don't have an edge
     b) Transaction costs (slippage model) are too aggressive
     c) Position sizing/risk management is destroying edge
     
  2. TRANSACTION COSTS: Check if gross PnL is positive but net is negative.
     If so, the edge exists but is being eaten by execution costs.
     
  3. YEARLY DECAY: If early years are profitable but later years are not,
     this suggests the strategy was overfit to historical data.
     
  4. SAMPLE SIZE: Micro-conditions with < 100 trades are not reliable.
     Even profitable-looking segments may be noise.
     
  RECOMMENDED NEXT STEPS:
  
  1. Check gross PnL (before slippage) - if positive, optimize execution
  2. Reduce slippage model aggressiveness and retest
  3. Focus on strategies with profit factor > 0.9 (closest to edge)
  4. Consider combining signals from multiple strategies (ensemble)
  5. Test with original restrictive regime filters to compare
""")


if __name__ == "__main__":
    main()
