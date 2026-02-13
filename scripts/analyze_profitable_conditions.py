#!/usr/bin/env python3
"""
Senior Quant Deep Dive: Analysis of 12 Profitable Conditions

For each profitable condition, analyze:
1. Statistical robustness (sample size, confidence intervals)
2. Win/loss distribution
3. Best/worst symbols
4. Holding period patterns
5. R-multiple distribution
6. Yearly consistency
7. Optimization opportunities
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict

RESULTS_DIR = Path("results")

# The 12 profitable conditions from the analysis
PROFITABLE_CONDITIONS = [
    ("gap_fill", "midday", "down", "normal", "risk_off"),
    ("orb", "opening", "flat", "high", "neutral"),
    ("orb", "opening", "down", "high", "risk_on"),
    ("vwap_trend_rider", "opening", "flat", "normal", "neutral"),
    ("orb", "opening", "down", "low", "risk_off"),
    ("orb", "opening", "flat", "normal", "risk_off"),
    ("vwap_trend_rider", "opening", "down", "low", "neutral"),
    ("vwap_reversion", "midday", "down", "high", "risk_on"),
    ("index_mean_reversion", "opening", "up", "high", "risk_on"),
    ("index_mean_reversion", "opening", "down", "high", "risk_off"),
    ("vwap_trend_rider", "opening", "up", "low", "neutral"),
    ("index_mean_reversion", "opening", "up", "normal", "risk_on"),
]


def load_trades(strategy: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"backtest_5yr_{strategy}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        trades = data.get("engine_trades", [])
        if isinstance(trades, list):
            return [t for t in trades if isinstance(t, dict)]
    return []


def filter_trades_by_condition(trades, session, trend, vol, risk):
    """Filter trades matching the regime condition."""
    matching = []
    for t in trades:
        regime = t.get("regime_tags_at_entry", {})
        if (
            regime.get("session") == session
            and regime.get("trend") == trend
            and regime.get("vol") == vol
            and regime.get("risk") == risk
        ):
            matching.append(t)
    return matching


def parse_year(entry_time):
    if isinstance(entry_time, str):
        return int(entry_time[:4])
    return 2020


def wilson_score_ci(wins, n, z=1.96):
    """Calculate Wilson score 95% CI for win rate."""
    if n == 0:
        return 0, 0
    p = wins / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0, center - margin), min(1, center + margin)


def analyze_condition(strategy, session, trend, vol, risk):
    """Deep analysis of a single profitable condition."""
    trades = load_trades(strategy)
    if not trades:
        return None

    filtered = filter_trades_by_condition(trades, session, trend, vol, risk)
    if not filtered:
        return None

    pnls = [t.get("pnl_net", 0) or 0 for t in filtered]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(filtered) if filtered else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Wilson CI
    ci_low, ci_high = wilson_score_ci(len(wins), len(filtered))

    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # R-multiples
    r_values = [t.get("pnl_r", 0) or 0 for t in filtered if t.get("pnl_r") is not None]
    avg_r = sum(r_values) / len(r_values) if r_values else 0

    # Holding period
    hold_times = [t.get("holding_period_seconds", 0) or 0 for t in filtered]
    avg_hold_mins = sum(hold_times) / len(hold_times) / 60 if hold_times else 0

    # Yearly breakdown
    yearly: DefaultDict[int, dict[str, float]] = defaultdict(lambda: {"trades": 0.0, "pnl": 0.0, "wins": 0.0})
    for t in filtered:
        year = parse_year(t.get("entry_time"))
        pnl = t.get("pnl_net", 0) or 0
        yearly[year]["trades"] += 1
        yearly[year]["pnl"] += pnl
        if pnl > 0:
            yearly[year]["wins"] += 1

    # Symbol breakdown
    symbols: DefaultDict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0.0, "pnl": 0.0, "wins": 0.0})
    for t in filtered:
        sym = t.get("symbol", "unknown")
        pnl = t.get("pnl_net", 0) or 0
        symbols[sym]["trades"] += 1
        symbols[sym]["pnl"] += pnl
        if pnl > 0:
            symbols[sym]["wins"] += 1

    # Sort symbols by PnL
    sorted_symbols = sorted(symbols.items(), key=lambda x: x[1]["pnl"], reverse=True)

    return {
        "strategy": strategy,
        "condition": f"{session}|{trend}|{vol}|{risk}",
        "trades": len(filtered),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "avg_r": avg_r,
        "avg_hold_mins": avg_hold_mins,
        "yearly": dict(yearly),
        "top_symbols": sorted_symbols[:5],
        "bottom_symbols": sorted_symbols[-3:] if len(sorted_symbols) > 3 else [],
        "profitable_years": sum(1 for y in yearly.values() if y["pnl"] > 0),
        "total_years": len(yearly),
    }


def print_deep_analysis(result):
    """Print detailed analysis for a condition."""
    if not result:
        return

    print(f"\n{'=' * 80}")
    print(f"  {result['strategy'].upper()} | {result['condition']}")
    print(f"{'=' * 80}")

    # Core stats
    print("\n  CORE METRICS:")
    print(f"    Trades: {result['trades']:,}")
    print(
        f"    Win Rate: {result['win_rate'] * 100:.1f}% (95% CI: {result['ci_low'] * 100:.1f}%-{result['ci_high'] * 100:.1f}%)"
    )
    print(f"    Total PnL: ${result['total_pnl']:,.0f}")
    print(f"    Expectancy: ${result['expectancy']:.2f}/trade")
    print(f"    Avg R: {result['avg_r']:.2f}")
    print(f"    Avg Hold: {result['avg_hold_mins']:.0f} min")

    # Win/Loss asymmetry
    print("\n  WIN/LOSS ASYMMETRY:")
    print(f"    Avg Win: ${result['avg_win']:.2f}")
    print(f"    Avg Loss: ${result['avg_loss']:.2f}")
    ratio = abs(result["avg_win"] / result["avg_loss"]) if result["avg_loss"] else 0
    print(f"    Win/Loss Ratio: {ratio:.2f}")

    # Statistical robustness
    print("\n  STATISTICAL ROBUSTNESS:")
    if result["trades"] >= 200:
        robustness = "HIGH ✅"
    elif result["trades"] >= 100:
        robustness = "MEDIUM ⚠️"
    else:
        robustness = "LOW ❌"
    print(f"    Sample Size: {robustness} ({result['trades']} trades)")

    ci_width = result["ci_high"] - result["ci_low"]
    if ci_width < 0.10:
        ci_quality = "TIGHT ✅"
    elif ci_width < 0.20:
        ci_quality = "MODERATE ⚠️"
    else:
        ci_quality = "WIDE ❌"
    print(f"    CI Width: {ci_quality} ({ci_width * 100:.1f}%)")

    # Yearly consistency
    print("\n  YEARLY CONSISTENCY:")
    print(f"    Profitable Years: {result['profitable_years']}/{result['total_years']}")
    for year in sorted(result["yearly"].keys()):
        y = result["yearly"][year]
        wr = y["wins"] / y["trades"] * 100 if y["trades"] > 0 else 0
        status = "✅" if y["pnl"] > 0 else "❌"
        print(f"    {status} {year}: {y['trades']:3} trades | WR={wr:5.1f}% | PnL=${y['pnl']:>8,.0f}")

    # Symbol concentration
    print("\n  TOP SYMBOLS:")
    for sym, stats in result["top_symbols"]:
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        print(f"    ✅ {sym:6} | {stats['trades']:3} trades | WR={wr:5.1f}% | PnL=${stats['pnl']:>7,.0f}")

    if result["bottom_symbols"]:
        print("\n  LOSING SYMBOLS (consider excluding):")
        for sym, stats in result["bottom_symbols"]:
            if stats["pnl"] < 0:
                wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
                print(f"    ❌ {sym:6} | {stats['trades']:3} trades | WR={wr:5.1f}% | PnL=${stats['pnl']:>7,.0f}")


def generate_optimization_recommendations(results):
    """Generate optimization recommendations for each condition."""
    print(f"\n{'=' * 80}")
    print("  OPTIMIZATION RECOMMENDATIONS BY CONDITION")
    print(f"{'=' * 80}")

    for r in results:
        if not r:
            continue

        print(f"\n  {r['strategy'].upper()} | {r['condition']}")
        print("  " + "-" * 60)

        recommendations = []

        # Sample size check
        if r["trades"] < 100:
            recommendations.append("LOW sample size - combine with similar regimes for robustness")

        # Yearly consistency
        if r["profitable_years"] < r["total_years"] * 0.6:
            recommendations.append(
                f"Only {r['profitable_years']}/{r['total_years']} years profitable - may be regime-dependent"
            )

        # Win/loss ratio
        ratio = abs(r["avg_win"] / r["avg_loss"]) if r["avg_loss"] else 0
        if ratio < 1.0 and r["win_rate"] < 0.5:
            recommendations.append("Poor risk/reward - consider widening targets or tightening stops")

        # CI width
        ci_width = r["ci_high"] - r["ci_low"]
        if ci_width > 0.15:
            recommendations.append("Wide confidence interval - need more trades for reliability")

        # Symbol concentration
        if r["top_symbols"]:
            top_sym_pnl = sum(s[1]["pnl"] for s in r["top_symbols"][:3])
            if top_sym_pnl > r["total_pnl"] * 0.8:
                recommendations.append(
                    f"Performance concentrated in top symbols - consider focusing on: {', '.join(s[0] for s in r['top_symbols'][:3])}"
                )

        # Exclude losing symbols
        if r["bottom_symbols"]:
            losers = [s[0] for s in r["bottom_symbols"] if s[1]["pnl"] < 0]
            if losers:
                recommendations.append(f"Consider excluding: {', '.join(losers)}")

        # Hold time optimization
        if r["avg_hold_mins"] < 30:
            recommendations.append("Short hold times - verify not getting stopped out too early")
        elif r["avg_hold_mins"] > 300:
            recommendations.append("Long hold times - consider tighter trailing stops")

        if recommendations:
            for rec in recommendations:
                print(f"    • {rec}")
        else:
            print("    ✅ No major issues - deploy with current settings")


def main():
    print("\n" + "=" * 80)
    print("  SENIOR QUANT DEEP DIVE: 12 PROFITABLE CONDITIONS")
    print("  5-Year Backtest Analysis (2020-2024)")
    print("=" * 80)

    results = []

    for strategy, session, trend, vol, risk in PROFITABLE_CONDITIONS:
        result = analyze_condition(strategy, session, trend, vol, risk)
        if result:
            print_deep_analysis(result)
            results.append(result)

    generate_optimization_recommendations(results)

    # Final ranking
    print(f"\n{'=' * 80}")
    print("  FINAL RANKING BY ROBUSTNESS (Expectancy × √Trades)")
    print(f"{'=' * 80}")

    for r in sorted(results, key=lambda x: x["expectancy"] * math.sqrt(x["trades"]), reverse=True):
        score = r["expectancy"] * math.sqrt(r["trades"])
        print(f"  {score:6.1f} | {r['strategy']:20} | {r['condition']}")
        print(f"         | {r['trades']:,} trades | E=${r['expectancy']:.2f} | WR={r['win_rate'] * 100:.1f}%")
        print()


if __name__ == "__main__":
    main()
