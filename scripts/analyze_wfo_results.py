#!/usr/bin/env python3
"""Analyze WFO results across all strategies and generate iteration report.

Usage:
    cd Cerberus
    uv run python scripts/analyze_wfo_results.py [--run-dir DIR]

Finds the latest WFO run for each strategy and produces a summary table
with recommendations for parameter adjustments.
"""

from __future__ import annotations

import json
from pathlib import Path

STRATEGIES = ["trend_rider_pro", "mean_reversion_pro", "orb_v2", "rsi_bounce", "momentum_fade"]
RUNS_DIR = Path("artifacts/optimization/runs")


def find_latest_results(strategy: str) -> dict | None:
    """Find the latest wfo_results.json for a strategy."""
    strat_dir = RUNS_DIR / strategy
    if not strat_dir.exists():
        return None

    # Find all run dirs with wfo_results.json, sorted by name (timestamp)
    results = sorted(strat_dir.glob("*/wfo_results.json"))
    if not results:
        return None

    with open(results[-1]) as f:
        data = json.load(f)
    data["_source"] = str(results[-1])
    return data


def analyze_strategy(data: dict) -> dict:
    """Extract key metrics from WFO results."""
    oos = data.get("oos_metrics", [])
    is_scores = data.get("is_scores", [])
    oos_scores = data.get("oos_scores", [])

    # Filter out hard-reject windows
    valid_oos = [m for m in oos if m.get("n_trades", 0) > 0]
    all_pnls = [m.get("net_pnl", 0) for m in oos]
    _valid_pnls = [m.get("net_pnl", 0) for m in valid_oos]  # noqa: F841

    analysis = {
        "wfo_efficiency": data.get("wfo_efficiency_ratio", 0),
        "n_windows": len(oos),
        "n_valid_windows": len(valid_oos),
        "n_profitable_windows": sum(1 for p in all_pnls if p > 0),
        "total_oos_pnl": sum(all_pnls),
        "avg_oos_trades": sum(m.get("n_trades", 0) for m in oos) / max(len(oos), 1),
        "avg_oos_winrate": sum(m.get("winrate", 0) for m in valid_oos) / max(len(valid_oos), 1),
        "avg_oos_pf": sum(m.get("profit_factor", 0) for m in valid_oos) / max(len(valid_oos), 1),
        "avg_oos_hold": sum(m.get("avg_hold_minutes", 0) for m in valid_oos) / max(len(valid_oos), 1),
        "avg_is_score": sum(is_scores) / max(len(is_scores), 1),
        "avg_oos_score": sum(s for s in oos_scores if s > -999) / max(sum(1 for s in oos_scores if s > -999), 1),
    }

    # Param stability
    stability = data.get("param_stability", {})
    analysis["stable_params"] = [p for p, v in stability.items() if v.get("stable", False)]
    analysis["unstable_params"] = [
        f"{p} (CV={v.get('cv', 0):.2f})"
        for p, v in stability.items()
        if not v.get("stable", False) and v.get("cv", 0) > 0.40
    ]
    analysis["borderline_params"] = [
        f"{p} (CV={v.get('cv', 0):.2f})"
        for p, v in stability.items()
        if not v.get("stable", False) and v.get("cv", 0) <= 0.40
    ]
    analysis["param_stability"] = stability

    return analysis


def recommend(name: str, a: dict) -> str:
    """Generate recommendation based on analysis."""
    if a["n_valid_windows"] == 0:
        return "FAILED — no trades generated"
    if a["wfo_efficiency"] >= 0.7 and a["n_profitable_windows"] >= a["n_windows"] * 0.5:
        return "DEPLOY"
    if a["wfo_efficiency"] >= 0.5 and a["n_profitable_windows"] >= 2:
        return "PAPER_TEST"
    if a["total_oos_pnl"] > 0 and a["n_profitable_windows"] >= 2:
        return "ITERATE — positive PnL, needs refinement"
    if a["avg_oos_trades"] > 30 and a["total_oos_pnl"] < 0:
        return "ITERATE — sufficient trades, negative PnL"
    return "NEEDS_WORK"


def main():
    print("=" * 80)
    print("WFO RESULTS ANALYSIS — CERBERUS AUTORESEARCH ITERATION")
    print("=" * 80)

    all_results = {}
    for strat in STRATEGIES:
        data = find_latest_results(strat)
        if data:
            all_results[strat] = (data, analyze_strategy(data))
        else:
            print(f"\n  {strat}: NO RESULTS FOUND")

    if not all_results:
        print("\nNo WFO results found. Run optimization first.")
        return

    # Summary table
    print(
        f"\n{'Strategy':<22s} {'WFO Eff':>8s} {'OOS PnL':>10s} {'Win/Total':>10s} {'Avg WR':>8s} {'Avg PF':>8s} {'Trades':>8s} {'Hold':>6s} {'Recommendation'}"
    )
    print("-" * 120)

    for strat in STRATEGIES:
        if strat not in all_results:
            print(f"{strat:<22s} {'—':>8s} {'—':>10s} {'—':>10s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>6s} NO DATA")
            continue
        _, a = all_results[strat]
        rec = recommend(strat, a)
        print(
            f"{strat:<22s} "
            f"{a['wfo_efficiency']:>8.3f} "
            f"${a['total_oos_pnl']:>9.0f} "
            f"{a['n_profitable_windows']}/{a['n_windows']:>8} "
            f"{a['avg_oos_winrate']:>7.1%} "
            f"{a['avg_oos_pf']:>8.2f} "
            f"{a['avg_oos_trades']:>8.0f} "
            f"{a['avg_oos_hold']:>5.0f}m "
            f"{rec}"
        )

    # Detailed per-strategy analysis
    for strat in STRATEGIES:
        if strat not in all_results:
            continue
        data, a = all_results[strat]
        print(f"\n{'=' * 60}")
        print(f"  {strat}")
        print(f"{'=' * 60}")
        print(f"  Source: {data['_source']}")
        print(f"  WFO Efficiency: {a['wfo_efficiency']:.3f}")
        print(f"  IS scores:  {[round(s, 2) for s in data.get('is_scores', [])]}")
        print(f"  OOS scores: {[round(s, 2) for s in data.get('oos_scores', [])]}")
        print(f"  Total OOS PnL: ${a['total_oos_pnl']:.2f}")

        oos = data.get("oos_metrics", [])
        if oos:
            print("\n  Per-Window OOS:")
            print(f"  {'Window':>6s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'PnL':>10s} {'Hold':>6s} {'Sharpe':>7s}")
            for i, m in enumerate(oos):
                print(
                    f"  {i + 1:>6d} "
                    f"{m.get('n_trades', 0):>7d} "
                    f"{m.get('winrate', 0):>5.1%} "
                    f"{m.get('profit_factor', 0):>6.2f} "
                    f"${m.get('net_pnl', 0):>9.2f} "
                    f"{m.get('avg_hold_minutes', 0):>5.0f}m "
                    f"{m.get('sharpe', 0):>7.2f}"
                )

        if a["param_stability"]:
            print("\n  Parameter Stability:")
            for p, v in a["param_stability"].items():
                status = "STABLE" if v.get("stable") else ("BORDERLINE" if v.get("cv", 0) <= 0.40 else "UNSTABLE")
                print(f"    {p:<25s} mean={v['mean']:.3f}  cv={v['cv']:.3f}  {status}")

        # Best params per window
        bpw = data.get("best_params_per_window", [])
        if bpw and any(bpw):
            print("\n  Best Params (window means):")
            all_keys = set()
            for bp in bpw:
                if bp:
                    all_keys.update(bp.keys())
            for k in sorted(all_keys):
                vals = [bp.get(k) for bp in bpw if bp and k in bp]
                if vals:
                    mean_v = sum(v for v in vals if isinstance(v, (int, float))) / max(len(vals), 1)
                    print(
                        f"    {k:<25s} mean={mean_v:.3f}  values={[round(v, 3) if isinstance(v, float) else v for v in vals]}"
                    )


if __name__ == "__main__":
    main()
