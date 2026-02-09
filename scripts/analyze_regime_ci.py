#!/usr/bin/env python3
"""
Enhanced Regime Analysis with Statistical Confidence Intervals.

Reads JSON trade results directly and calculates Wilson score intervals
for reliable win rate estimation with sample size awareness.

Usage:
    python scripts/analyze_regime_ci.py artifacts/backtests/6mo_all_strategies
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class RegimeBucket:
    """Statistics for a strategy+regime combination."""

    strategy: str
    regime_axis: str
    regime_value: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total_trades if self.total_trades else 0.0

    def wilson_ci(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Wilson score interval for win rate (more accurate for small samples)."""
        n = self.total_trades
        if n == 0:
            return (0.0, 0.0)
        p = self.win_rate
        z = 1.96  # 95% CI
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        return (max(0.0, center - spread), min(1.0, center + spread))

    def confidence_rating(self) -> str:
        if self.total_trades < 10:
            return "⚠️ insufficient"
        if self.total_trades < 30:
            return "🔸 low"
        if self.total_trades < 100:
            return "🔹 medium"
        return "✅ high"


def load_trades(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f).get("engine_trades", [])


def analyze(trades: List[Dict]) -> Dict[Tuple[str, str, str], RegimeBucket]:
    buckets: Dict[Tuple[str, str, str], RegimeBucket] = {}
    for t in trades:
        strat = t.get("strategy", "unknown")
        tags = t.get("regime_tags_at_entry", {}) or t.get("regime_tags", {})
        pnl = float(t.get("pnl_net", 0) or 0)
        for axis in ["trend", "vol", "session", "risk"]:
            val = tags.get(axis, "unknown")
            key = (strat, axis, val)
            if key not in buckets:
                buckets[key] = RegimeBucket(strat, axis, val)
            b = buckets[key]
            b.total_trades += 1
            b.total_pnl += pnl
            if pnl > 0:
                b.winning_trades += 1
    return buckets


def report(buckets: Dict) -> None:
    print("=" * 100)
    print("REGIME ANALYSIS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 100)

    by_strat = defaultdict(list)
    for b in buckets.values():
        by_strat[b.strategy].append(b)

    for strat in sorted(by_strat):
        bs = by_strat[strat]
        total = sum(b.total_trades for b in bs) // 4
        pnl = sum(b.total_pnl for b in bs) // 4
        print(
            f"\n{'─' * 80}\n{strat.upper()} | {total:,} trades | ${pnl:,.0f} PnL\n{'─' * 80}"
        )

        for axis in ["session", "trend", "vol", "risk"]:
            axis_bs = sorted(
                [b for b in bs if b.regime_axis == axis],
                key=lambda x: x.wilson_ci()[0],
                reverse=True,
            )
            if not axis_bs:
                continue
            print(f"\n  {axis.upper()}:")
            print(
                f"  {'Value':<12} {'N':>6} {'WR':>8} {'95% CI':>20} {'AvgPnL':>12} {'Conf'}"
            )
            for b in axis_bs:
                lo, hi = b.wilson_ci()
                sig = "★" if lo > 0.20 else ""
                print(
                    f"  {b.regime_value:<12} {b.total_trades:>6} {b.win_rate * 100:>7.1f}% "
                    f"[{lo * 100:>5.1f}% - {hi * 100:>5.1f}%] ${b.avg_pnl:>10,.2f} {b.confidence_rating()} {sig}"
                )

    # Recommendations
    print(
        f"\n{'=' * 100}\nHIGH-CONFIDENCE TARGETS (95% CI lower > 25%, N≥20)\n{'=' * 100}"
    )
    good = [
        (b, b.wilson_ci()[0])
        for b in buckets.values()
        if b.total_trades >= 20 and b.wilson_ci()[0] >= 0.25
    ]
    good.sort(key=lambda x: x[1], reverse=True)

    if good:
        print(f"\n{'Strategy':<22} {'Regime':<18} {'N':>6} {'WR':>8} {'CI Low':>8}")
        print("-" * 70)
        for b, lo in good[:15]:
            print(
                f"{b.strategy:<22} {b.regime_axis}={b.regime_value:<12} "
                f"{b.total_trades:>6} {b.win_rate * 100:>7.1f}% {lo * 100:>7.1f}%"
            )
    else:
        print(
            "\nNo regime targets meet threshold. Consider longer backtest or lower threshold."
        )

    print("\n★ = CI lower bound > 20% (statistically significant edge)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_regime_ci.py <results.json>")
        sys.exit(1)
    path = sys.argv[1]
    if not Path(path).exists():
        print(f"Not found: {path}")
        sys.exit(1)
    trades = load_trades(path)
    print(f"Loaded {len(trades):,} trades from {path}\n")
    report(analyze(trades))


if __name__ == "__main__":
    main()
