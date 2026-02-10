#!/usr/bin/env python3
"""
Analyze backtest results by market regime to find profitable conditions.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

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


def load_trades(strategy: str) -> List[Dict[str, Any]]:
    """Load trades from backtest result file."""
    path = RESULTS_DIR / f"backtest_5yr_{strategy}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("engine_trades", [])


def analyze_by_regime(trades: List[Dict], axis: str) -> Dict[str, Dict]:
    """Group trades by a specific regime axis and calculate stats."""
    groups = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_list": []})

    for t in trades:
        regime_tags = t.get("regime_tags_at_entry", {})
        value = regime_tags.get(axis, "unknown")
        pnl = t.get("pnl_net", 0) or 0

        groups[value]["trades"] += 1
        groups[value]["pnl"] += pnl
        groups[value]["pnl_list"].append(pnl)
        if pnl > 0:
            groups[value]["wins"] += 1

    # Calculate derived stats
    for v in groups.values():
        v["win_rate"] = v["wins"] / v["trades"] if v["trades"] > 0 else 0
        v["avg_pnl"] = v["pnl"] / v["trades"] if v["trades"] > 0 else 0

    return dict(groups)


def analyze_by_combination(trades: List[Dict], axes: List[str]) -> Dict[str, Dict]:
    """Group trades by combination of regime axes."""
    groups = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})

    for t in trades:
        regime_tags = t.get("regime_tags_at_entry", {})
        key = tuple(regime_tags.get(a, "unknown") for a in axes)
        key_str = " | ".join(f"{a}={v}" for a, v in zip(axes, key, strict=False))
        pnl = t.get("pnl_net", 0) or 0

        groups[key_str]["trades"] += 1
        groups[key_str]["pnl"] += pnl
        if pnl > 0:
            groups[key_str]["wins"] += 1

    for v in groups.values():
        v["win_rate"] = v["wins"] / v["trades"] if v["trades"] > 0 else 0
        v["avg_pnl"] = v["pnl"] / v["trades"] if v["trades"] > 0 else 0

    return dict(groups)


def print_regime_analysis(strategy: str, trades: List[Dict]):
    """Print regime analysis for a strategy."""
    print(f"\n{'=' * 70}")
    print(f"  {strategy.upper()} - {len(trades):,} trades")
    print(f"{'=' * 70}")

    # Overall stats
    total_pnl = sum(t.get("pnl_net", 0) or 0 for t in trades)
    wins = sum(1 for t in trades if (t.get("pnl_net", 0) or 0) > 0)
    print(f"  Overall: PnL=${total_pnl:,.0f} | WR={wins / len(trades) * 100:.1f}%")

    # By each regime axis
    for axis in ["session", "trend", "vol", "risk"]:
        print(f"\n  By {axis.upper()}:")
        results = analyze_by_regime(trades, axis)

        # Sort by PnL descending
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["pnl"], reverse=True
        )

        for value, stats in sorted_results:
            profitable = "✅" if stats["pnl"] > 0 else "❌"
            print(
                f"    {profitable} {value:15} | {stats['trades']:6,} trades | "
                f"WR={stats['win_rate'] * 100:5.1f}% | PnL=${stats['pnl']:>10,.0f} | "
                f"Avg=${stats['avg_pnl']:>7,.2f}"
            )

    # Best combinations (session + trend)
    print("\n  BEST COMBINATIONS (session + trend + vol):")
    combo_results = analyze_by_combination(trades, ["session", "trend", "vol"])

    # Filter profitable and sort
    profitable_combos = [
        (k, v) for k, v in combo_results.items() if v["pnl"] > 0 and v["trades"] >= 50
    ]  # Min 50 trades
    profitable_combos.sort(key=lambda x: x[1]["pnl"], reverse=True)

    if profitable_combos:
        for key, stats in profitable_combos[:10]:  # Top 10
            print(f"    ✅ {key}")
            print(
                f"       {stats['trades']:,} trades | WR={stats['win_rate'] * 100:.1f}% | "
                f"PnL=${stats['pnl']:,.0f}"
            )
    else:
        print("    No profitable combinations with >= 50 trades found")


def find_all_profitable_regimes():
    """Find all profitable regime combinations across strategies."""
    print("\n" + "=" * 70)
    print("  SUMMARY: PROFITABLE REGIME COMBINATIONS")
    print("=" * 70)

    profitable = []

    for strategy in STRATEGIES:
        trades = load_trades(strategy)
        if not trades:
            continue

        combo_results = analyze_by_combination(trades, ["session", "trend", "vol"])

        for key, stats in combo_results.items():
            if stats["pnl"] > 0 and stats["trades"] >= 100:
                profitable.append(
                    {
                        "strategy": strategy,
                        "regime": key,
                        "trades": stats["trades"],
                        "pnl": stats["pnl"],
                        "win_rate": stats["win_rate"],
                    }
                )

    # Sort by PnL
    profitable.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"\nFound {len(profitable)} profitable combinations (>= 100 trades):\n")
    for p in profitable[:20]:  # Top 20
        print(f"  ✅ {p['strategy']:20} | {p['regime']}")
        print(
            f"     {p['trades']:,} trades | WR={p['win_rate'] * 100:.1f}% | PnL=${p['pnl']:,.0f}"
        )
        print()


def main():
    print("\n" + "=" * 70)
    print("  5-YEAR BACKTEST REGIME ANALYSIS (2020-2024)")
    print("=" * 70)

    for strategy in STRATEGIES:
        trades = load_trades(strategy)
        if trades:
            print_regime_analysis(strategy, trades)

    find_all_profitable_regimes()


if __name__ == "__main__":
    main()
