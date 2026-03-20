"""Post-backtest diagnostics engine.

Identifies performance leaks across strategies, regimes, time slots,
and exit types to guide parameter tuning and strategy selection.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class StrategyRanking:
    strategy: str
    net_pnl: float
    n_trades: int
    win_rate: float


@dataclass
class RegimeMismatch:
    strategy: str
    regime_axis: str
    regime_value: str
    avg_pnl: float
    n_trades: int
    recommendation: str


@dataclass
class TimeSlotEdge:
    hour: int
    avg_pnl: float
    n_trades: int
    win_rate: float


@dataclass
class HoldAnalysis:
    exit_type: str
    avg_pnl: float
    n_trades: int


@dataclass
class DiagnosticsReport:
    strategy_rankings: list[StrategyRanking]
    regime_mismatches: list[RegimeMismatch]
    time_edge_map: dict[str, list[TimeSlotEdge]]
    hold_analysis: dict[str, list[HoldAnalysis]]
    summary: str


def _compute_strategy_rankings(trades: list[dict]) -> list[StrategyRanking]:
    """Group by strategy, compute net_pnl / n_trades / win_rate, sort by net_pnl desc."""
    groups: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        groups[t["strategy"]].append(t["pnl"])

    rankings = []
    for strategy, pnls in groups.items():
        n = len(pnls)
        net = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        rankings.append(StrategyRanking(strategy=strategy, net_pnl=net, n_trades=n, win_rate=wins / n if n else 0.0))

    rankings.sort(key=lambda r: r.net_pnl, reverse=True)
    return rankings


def _detect_regime_mismatches(trades: list[dict]) -> list[RegimeMismatch]:
    """Flag (strategy, regime_trend) groups with n_trades >= 3 and avg_pnl < 0."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for t in trades:
        groups[(t["strategy"], t["regime_trend"])].append(t["pnl"])

    mismatches = []
    for (strategy, regime_value), pnls in groups.items():
        n = len(pnls)
        if n < 3:
            continue
        avg = sum(pnls) / n
        if avg < 0:
            mismatches.append(
                RegimeMismatch(
                    strategy=strategy,
                    regime_axis="trend",
                    regime_value=regime_value,
                    avg_pnl=avg,
                    n_trades=n,
                    recommendation=f"Consider removing {regime_value} from activation.trend",
                )
            )
    return mismatches


def _compute_time_edge(trades: list[dict]) -> dict[str, list[TimeSlotEdge]]:
    """Group by (strategy, entry_hour), compute avg_pnl and win_rate per slot."""
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for t in trades:
        groups[(t["strategy"], t["entry_hour"])].append(t["pnl"])

    result: dict[str, list[TimeSlotEdge]] = defaultdict(list)
    for (strategy, hour), pnls in groups.items():
        n = len(pnls)
        avg = sum(pnls) / n
        wins = sum(1 for p in pnls if p > 0)
        result[strategy].append(TimeSlotEdge(hour=hour, avg_pnl=avg, n_trades=n, win_rate=wins / n if n else 0.0))

    for slots in result.values():
        slots.sort(key=lambda s: s.hour)

    return dict(result)


def _compute_hold_analysis(trades: list[dict]) -> dict[str, list[HoldAnalysis]]:
    """Group by (strategy, exit_type), compute avg_pnl per exit type."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for t in trades:
        groups[(t["strategy"], t["exit_type"])].append(t["pnl"])

    result: dict[str, list[HoldAnalysis]] = defaultdict(list)
    for (strategy, exit_type), pnls in groups.items():
        n = len(pnls)
        avg = sum(pnls) / n
        result[strategy].append(HoldAnalysis(exit_type=exit_type, avg_pnl=avg, n_trades=n))

    for items in result.values():
        items.sort(key=lambda h: h.exit_type)

    return dict(result)


def _generate_summary(
    rankings: list[StrategyRanking],
    mismatches: list[RegimeMismatch],
    time_edge_map: dict[str, list[TimeSlotEdge]],
) -> str:
    """Generate a 2-3 sentence plain text summary."""
    parts: list[str] = []

    if rankings:
        best = rankings[0]
        parts.append(f"Top performer: {best.strategy} (${best.net_pnl:.0f}).")

    if mismatches:
        strats = sorted({m.strategy for m in mismatches})
        regimes = sorted({m.regime_value for m in mismatches})
        parts.append(f"Regime mismatches detected for {', '.join(strats)} in {', '.join(regimes)} regimes.")

    # Find strongest time edge (best avg_pnl slot across all strategies)
    best_slot = None
    best_slot_strategy = ""
    for strategy, slots in time_edge_map.items():
        for slot in slots:
            if slot.n_trades >= 2 and (best_slot is None or slot.avg_pnl > best_slot.avg_pnl):
                best_slot = slot
                best_slot_strategy = strategy
    if best_slot is not None:
        parts.append(f"Best time edge: {best_slot_strategy} at hour {best_slot.hour} (avg ${best_slot.avg_pnl:.0f}).")

    return " ".join(parts) if parts else "Insufficient data for diagnostics summary."


def run_diagnostics(trades: list[dict], min_trades: int = 20) -> DiagnosticsReport:
    """Run full post-backtest diagnostics suite.

    Args:
        trades: List of trade dicts with keys: strategy, pnl, regime_trend,
                entry_hour, hold_minutes, exit_type.
        min_trades: Minimum total trades required (used for gating; analyses
                    still run on whatever data is available).

    Returns:
        DiagnosticsReport with all analyses populated.
    """
    rankings = _compute_strategy_rankings(trades)
    mismatches = _detect_regime_mismatches(trades)
    time_edge = _compute_time_edge(trades)
    hold = _compute_hold_analysis(trades)
    summary = _generate_summary(rankings, mismatches, time_edge)

    return DiagnosticsReport(
        strategy_rankings=rankings,
        regime_mismatches=mismatches,
        time_edge_map=time_edge,
        hold_analysis=hold,
        summary=summary,
    )
