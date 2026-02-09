#!/usr/bin/env python3
"""
Quick deployment analysis specifically for low-slippage results.
"""

import json
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
    path = RESULTS_DIR / f"backtest_5yr_low_slip_{strategy}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("engine_trades", [])


def analyze_all_conditions(trades):
    """Get all session×regime combinations."""
    combos = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_gross": 0.0})

    for t in trades:
        regime = t.get("regime_tags_at_entry", {})
        key = (
            regime.get("session"),
            regime.get("trend"),
            regime.get("vol"),
            regime.get("risk"),
        )
        pnl = t.get("pnl_net", 0) or 0
        pnl_gross = t.get("pnl_gross", 0) or 0

        combos[key]["trades"] += 1
        combos[key]["pnl"] += pnl
        combos[key]["pnl_gross"] += pnl_gross
        if pnl > 0:
            combos[key]["wins"] += 1

    return dict(combos)


def main():
    print("\n" + "=" * 80)
    print("  LOW-SLIPPAGE DEPLOYMENT ANALYSIS")
    print("  Slippage: 2.0 bps (reduced from 5.0)")
    print("=" * 80)

    all_profitable = []

    for strategy in STRATEGIES:
        trades = load_trades(strategy)
        if not trades:
            continue

        total_pnl = sum(t.get("pnl_net", 0) or 0 for t in trades)
        total_gross = sum(t.get("pnl_gross", 0) or 0 for t in trades)
        wins = sum(1 for t in trades if (t.get("pnl_net", 0) or 0) > 0)

        print(f"\n{'=' * 60}")
        print(f"  {strategy.upper()}")
        print(
            f"  {len(trades):,} trades | WR={wins / len(trades) * 100:.1f}% | Net=${total_pnl:,.0f} | Gross=${total_gross:,.0f}"
        )
        print(f"{'=' * 60}")

        combos = analyze_all_conditions(trades)

        # Find profitable conditions
        profitable = [
            (k, v) for k, v in combos.items() if v["pnl"] > 0 and v["trades"] >= 50
        ]
        profitable.sort(key=lambda x: x[1]["pnl"], reverse=True)

        if profitable:
            print("\n  PROFITABLE CONDITIONS (>= 50 trades):")
            for key, stats in profitable[:10]:
                session, trend, vol, risk = key
                wr = stats["wins"] / stats["trades"] * 100
                avg_pnl = stats["pnl"] / stats["trades"]
                print(
                    f"    ✅ session={session} | trend={trend} | vol={vol} | risk={risk}"
                )
                print(
                    f"       {stats['trades']} trades | WR={wr:.1f}% | PnL=${stats['pnl']:,.0f} | Avg=${avg_pnl:.2f}"
                )

                all_profitable.append(
                    {
                        "strategy": strategy,
                        "session": session,
                        "trend": trend,
                        "vol": vol,
                        "risk": risk,
                        "trades": stats["trades"],
                        "pnl": stats["pnl"],
                        "win_rate": wr,
                        "avg_pnl": avg_pnl,
                    }
                )
        else:
            print("\n  No profitable conditions with >= 50 trades")

    # Summary
    print("\n" + "=" * 80)
    print("  FINAL DEPLOYMENT RECOMMENDATIONS (Low Slippage)")
    print("=" * 80)

    if all_profitable:
        all_profitable.sort(key=lambda x: x["pnl"], reverse=True)
        print(f"\n  Found {len(all_profitable)} PROFITABLE conditions:\n")

        for p in all_profitable[:20]:
            print(f"  ✅ {p['strategy'].upper()}")
            print(
                f"     Session: {p['session']} | Trend: {p['trend']} | Vol: {p['vol']} | Risk: {p['risk']}"
            )
            print(
                f"     {p['trades']} trades | WR={p['win_rate']:.1f}% | PnL=${p['pnl']:,.0f} | Avg=${p['avg_pnl']:.2f}/trade"
            )
            print()
    else:
        print("\n  ⚠️  NO profitable conditions found even with reduced slippage")


if __name__ == "__main__":
    main()
