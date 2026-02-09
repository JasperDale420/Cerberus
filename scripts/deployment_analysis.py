#!/usr/bin/env python3
"""
Strategy Deployment Analysis
Analyze each strategy by:
- Session (opening, midday, power_hour, close)
- Trend (up, down, flat)
- Volatility (low, normal, high, shock)
- Risk (risk_on, neutral, risk_off)

Creates a matrix to identify optimal deployment conditions.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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


def load_trades(strategy: str) -> List[Dict]:
    path = RESULTS_DIR / f"backtest_5yr_{strategy}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("engine_trades", [])


def analyze_session_regime_matrix(trades: List[Dict]) -> Dict:
    """Create session × regime matrix."""
    # session × trend matrix
    matrix = defaultdict(
        lambda: defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_gross": 0.0}
        )
    )

    for t in trades:
        regime = t.get("regime_tags_at_entry", {})
        session = regime.get("session", "unknown")
        trend = regime.get("trend", "unknown")
        vol = regime.get("vol", "unknown")
        risk = regime.get("risk", "unknown")

        pnl = t.get("pnl_net", 0) or 0
        pnl_gross = t.get("pnl_gross", 0) or 0

        # Session × Trend
        key = f"{session}|{trend}"
        matrix["session_trend"][key]["trades"] += 1
        matrix["session_trend"][key]["pnl"] += pnl
        matrix["session_trend"][key]["pnl_gross"] += pnl_gross
        if pnl > 0:
            matrix["session_trend"][key]["wins"] += 1

        # Session × Vol
        key = f"{session}|{vol}"
        matrix["session_vol"][key]["trades"] += 1
        matrix["session_vol"][key]["pnl"] += pnl
        matrix["session_vol"][key]["pnl_gross"] += pnl_gross
        if pnl > 0:
            matrix["session_vol"][key]["wins"] += 1

        # Session × Risk
        key = f"{session}|{risk}"
        matrix["session_risk"][key]["trades"] += 1
        matrix["session_risk"][key]["pnl"] += pnl
        matrix["session_risk"][key]["pnl_gross"] += pnl_gross
        if pnl > 0:
            matrix["session_risk"][key]["wins"] += 1

        # Full combination
        key = f"{session}|{trend}|{vol}|{risk}"
        matrix["full"][key]["trades"] += 1
        matrix["full"][key]["pnl"] += pnl
        matrix["full"][key]["pnl_gross"] += pnl_gross
        if pnl > 0:
            matrix["full"][key]["wins"] += 1

    return dict(matrix)


def print_matrix(name: str, data: Dict, row_labels: List[str], col_labels: List[str]):
    """Print a 2D matrix with headers."""
    # Calculate win rate and mark profitable cells
    print(f"\n  {name}:")

    # Header
    header = "              "
    for col in col_labels:
        header += f"{col:>12}"
    print(header)
    print("  " + "-" * (14 + 12 * len(col_labels)))

    for row in row_labels:
        line = f"  {row:>12} |"
        for col in col_labels:
            key = f"{row}|{col}"
            stats = data.get(key, {"trades": 0, "pnl": 0, "wins": 0})
            if stats["trades"] >= 20:  # Minimum sample
                wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
                pnl = stats["pnl"]
                if pnl > 0:
                    cell = f"✅{pnl:>7.0f}"
                else:
                    cell = f"  {pnl:>7.0f}"
            else:
                cell = "       -"
            line += f"{cell:>12}"
        print(line)


def analyze_strategy(strategy: str, trades: List[Dict]):
    """Full analysis for one strategy."""
    if not trades:
        return

    total_pnl = sum(t.get("pnl_net", 0) or 0 for t in trades)
    total_pnl_gross = sum(t.get("pnl_gross", 0) or 0 for t in trades)
    wins = sum(1 for t in trades if (t.get("pnl_net", 0) or 0) > 0)

    print(f"\n{'=' * 80}")
    print(f"  {strategy.upper()} - DEPLOYMENT ANALYSIS")
    print(
        f"  {len(trades):,} trades | WR={wins / len(trades) * 100:.1f}% | Net=${total_pnl:,.0f} | Gross=${total_pnl_gross:,.0f}"
    )
    print(f"{'=' * 80}")

    matrix = analyze_session_regime_matrix(trades)

    # Session × Trend matrix
    sessions = ["opening", "midday", "power_hour", "close"]
    trends = ["up", "down", "flat"]
    print_matrix("SESSION × TREND (Net PnL)", matrix["session_trend"], sessions, trends)

    # Session × Vol matrix
    vols = ["low", "normal", "high", "shock"]
    print_matrix(
        "SESSION × VOLATILITY (Net PnL)", matrix["session_vol"], sessions, vols
    )

    # Session × Risk matrix
    risks = ["risk_on", "neutral", "risk_off"]
    print_matrix(
        "SESSION × RISK SENTIMENT (Net PnL)", matrix["session_risk"], sessions, risks
    )

    # Find profitable conditions (min 50 trades for reliability)
    print("\n  PROFITABLE CONDITIONS (>= 50 trades):")
    profitable = []
    for key, stats in matrix["full"].items():
        if stats["trades"] >= 50 and stats["pnl"] > 0:
            wr = stats["wins"] / stats["trades"] * 100
            profitable.append((key, stats["trades"], stats["pnl"], wr))

    profitable.sort(key=lambda x: x[2], reverse=True)  # Sort by PnL

    if profitable:
        for key, trades_n, pnl, wr in profitable[:10]:
            parts = key.split("|")
            print(
                f"    ✅ session={parts[0]} | trend={parts[1]} | vol={parts[2]} | risk={parts[3]}"
            )
            print(f"       {trades_n} trades | WR={wr:.1f}% | PnL=${pnl:,.0f}")
    else:
        print("    None found with >= 50 trades")

    # Find conditions that are profitable on GROSS basis (edge exists but slippage kills it)
    print("\n  GROSS-PROFITABLE CONDITIONS (edge exists, slippage kills it):")
    gross_profitable = []
    for key, stats in matrix["full"].items():
        if stats["trades"] >= 50 and stats["pnl_gross"] > 0 and stats["pnl"] <= 0:
            gross_profitable.append(
                (key, stats["trades"], stats["pnl_gross"], stats["pnl"])
            )

    gross_profitable.sort(key=lambda x: x[2], reverse=True)

    if gross_profitable:
        for key, trades_n, pnl_gross, pnl_net in gross_profitable[:5]:
            parts = key.split("|")
            print(
                f"    ⚠️  session={parts[0]} | trend={parts[1]} | vol={parts[2]} | risk={parts[3]}"
            )
            print(
                f"       {trades_n} trades | Gross=${pnl_gross:,.0f} | Net=${pnl_net:,.0f} (slippage impact)"
            )
    else:
        print("    None found")


def create_deployment_recommendations():
    """Create final deployment recommendations."""
    print("\n" + "=" * 80)
    print("  DEPLOYMENT RECOMMENDATIONS")
    print("=" * 80)

    all_profitable = []

    for strategy in STRATEGIES:
        trades = load_trades(strategy)
        if not trades:
            continue

        matrix = analyze_session_regime_matrix(trades)

        for key, stats in matrix["full"].items():
            if stats["trades"] >= 100 and stats["pnl"] > 0:
                wr = stats["wins"] / stats["trades"] * 100
                avg_pnl = stats["pnl"] / stats["trades"]
                parts = key.split("|")
                all_profitable.append(
                    {
                        "strategy": strategy,
                        "session": parts[0],
                        "trend": parts[1],
                        "vol": parts[2],
                        "risk": parts[3],
                        "trades": stats["trades"],
                        "pnl": stats["pnl"],
                        "win_rate": wr,
                        "avg_pnl": avg_pnl,
                    }
                )

    if all_profitable:
        all_profitable.sort(key=lambda x: x["pnl"], reverse=True)
        print(
            f"\n  Found {len(all_profitable)} PROFITABLE conditions with >= 100 trades:\n"
        )

        for p in all_profitable[:15]:
            print(f"  ✅ {p['strategy'].upper()}")
            print(
                f"     Session: {p['session']} | Trend: {p['trend']} | Vol: {p['vol']} | Risk: {p['risk']}"
            )
            print(
                f"     {p['trades']} trades | WR={p['win_rate']:.1f}% | PnL=${p['pnl']:,.0f} | Avg=${p['avg_pnl']:.2f}/trade"
            )
            print()
    else:
        print("\n  ⚠️  NO conditions with >= 100 trades are profitable")
        print("     Consider:")
        print("     1. Reducing slippage/commission assumptions")
        print("     2. Widening regime filters to get more sample size")
        print("     3. These strategies may not have a tradeable edge")


def main():
    print("\n" + "=" * 80)
    print("  STRATEGY DEPLOYMENT ANALYSIS - BY SESSION × REGIME")
    print("  Using 5-year backtest data (2020-2024)")
    print("=" * 80)

    for strategy in STRATEGIES:
        trades = load_trades(strategy)
        if trades:
            analyze_strategy(strategy, trades)

    create_deployment_recommendations()


if __name__ == "__main__":
    main()
