"""Per-regime performance statistics for backtest report cards.

Groups trades by regime labels (trend, volatility, liquidity, etc.) and
enrichment flags (near_earnings, near_fomc, opex_week) to show how a
strategy performs under different market conditions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Regime fields written by Phase 1.2 (entry enrichment)
# ---------------------------------------------------------------------------

REGIME_TREND_FIELD = "entry_regime_trend"
REGIME_VOL_FIELD = "entry_regime_vol"
REGIME_LIQUIDITY_FIELD = "entry_liquidity"
REGIME_CORRELATION_FIELD = "entry_correlation"
SESSION_PHASE_FIELD = "entry_session_phase"

ENRICHMENT_BOOL_FIELDS = ("entry_near_earnings", "entry_near_fomc", "entry_opex_week")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if abs(denom) > 1e-12 else default


def _group_stats(pnls: list[float]) -> dict[str, Any]:
    """Compute core performance stats for a list of trade PnLs."""
    n = len(pnls)
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / n
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = _safe_div(gross_profit, gross_loss)
    avg_pnl = total_pnl / n

    # Simple drawdown from cumulative PnL curve
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = _safe_div(peak - cum, abs(peak) if abs(peak) > 1e-12 else 1.0)
        if dd > max_dd:
            max_dd = dd

    # Sharpe and Sortino (not annualized -- per-trade basis)
    if n > 1:
        mean_r = avg_pnl
        variance = sum((p - mean_r) ** 2 for p in pnls) / (n - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = _safe_div(mean_r, std_r)

        downside_sq = sum(min(p, 0.0) ** 2 for p in pnls) / n
        downside_dev = math.sqrt(downside_sq)
        sortino = _safe_div(mean_r, downside_dev)
    else:
        sharpe = 0.0
        sortino = 0.0

    return {
        "n_trades": n,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_pnl": round(avg_pnl, 4),
        "total_pnl": round(total_pnl, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 4) if math.isfinite(sharpe) else 0.0,
        "sortino": round(sortino, 4) if math.isfinite(sortino) else 0.0,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_regime_breakdown(
    trades: list[dict],
    regime_field: str = "entry_regime_trend",
    pnl_key: str = "pnl",
) -> dict[str, dict]:
    """Group trades by a regime field and compute per-group statistics.

    Returns:
        {
            "UP": {"n_trades": 42, "win_rate": 0.55, "profit_factor": 1.23,
                   "avg_pnl": 12.50, "total_pnl": 525.0, "max_drawdown": 0.03,
                   "sharpe": 1.2, "sortino": 1.8},
            "FLAT": {...},
            "DOWN": {...},
        }

    If *regime_field* is missing from all trades, returns an empty dict.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    found_any = False

    for t in trades:
        regime = t.get(regime_field)
        if regime is None:
            continue
        found_any = True
        groups[str(regime)].append(float(t.get(pnl_key, 0.0)))

    if not found_any:
        return {}

    return {regime: _group_stats(pnls) for regime, pnls in sorted(groups.items())}


def compute_regime_matrix(
    trades: list[dict],
    row_field: str = "entry_regime_trend",
    col_field: str = "entry_regime_vol",
    pnl_key: str = "pnl",
) -> dict[str, Any]:
    """2D breakdown: trend x vol regime matrix.

    Returns:
        {
            "rows": ["DOWN", "FLAT", "UP"],
            "cols": ["HIGH", "LOW", "NORMAL"],
            "cells": {
                "UP|NORMAL": {"n_trades": 15, "win_rate": 0.60, "avg_pnl": 8.5},
                ...
            },
        }

    If either field is missing from all trades, returns an empty dict.
    """
    cells: dict[str, list[float]] = defaultdict(list)
    row_values: set[str] = set()
    col_values: set[str] = set()
    found_any = False

    for t in trades:
        row_val = t.get(row_field)
        col_val = t.get(col_field)
        if row_val is None or col_val is None:
            continue
        found_any = True
        row_str = str(row_val)
        col_str = str(col_val)
        row_values.add(row_str)
        col_values.add(col_str)
        cells[f"{row_str}|{col_str}"].append(float(t.get(pnl_key, 0.0)))

    if not found_any:
        return {}

    result_cells: dict[str, dict[str, Any]] = {}
    for key, pnls in cells.items():
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        result_cells[key] = {
            "n_trades": n,
            "win_rate": round(wins / n, 4) if n > 0 else 0.0,
            "avg_pnl": round(sum(pnls) / n, 4) if n > 0 else 0.0,
            "total_pnl": round(sum(pnls), 4),
        }

    return {
        "rows": sorted(row_values),
        "cols": sorted(col_values),
        "cells": result_cells,
    }


VALID_SESSION_PHASES = ("opening_15m", "morning", "midday", "power_hour", "close_15m")


def compute_session_phase_stats(
    trades: list[dict],
    pnl_key: str = "pnl",
    hold_minutes_key: str = "hold_minutes",
) -> dict[str, Any]:
    """Break down strategy performance by intraday session phase.

    Expects trades to carry ``entry_session_phase`` with values from
    ``VALID_SESSION_PHASES``.  Trades missing the field or with unrecognised
    values are grouped under ``"unknown"``.

    Returns a dict keyed by phase with core stats plus ``best_trade``,
    ``worst_trade``, and ``avg_hold_minutes``.  Top-level summary keys:

    - ``best_session``  — phase with highest avg_pnl
    - ``worst_session`` — phase with lowest avg_pnl
    - ``recommended_sessions`` — phases with profit_factor > 1.2
    - ``avoid_sessions`` — phases with 0 < profit_factor < 0.8
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for t in trades:
        phase = t.get(SESSION_PHASE_FIELD)
        if phase is None or phase not in VALID_SESSION_PHASES:
            groups["unknown"].append(t)
        else:
            groups[str(phase)].append(t)

    result: dict[str, Any] = {}

    for phase in [*VALID_SESSION_PHASES, "unknown"]:
        phase_trades = groups.get(phase, [])
        pnls = [float(t.get(pnl_key, 0.0)) for t in phase_trades]
        base = _group_stats(pnls)

        # Extra fields specific to session phase stats
        base["best_trade"] = round(max(pnls), 4) if pnls else 0.0
        base["worst_trade"] = round(min(pnls), 4) if pnls else 0.0

        hold_vals = [float(t[hold_minutes_key]) for t in phase_trades if hold_minutes_key in t]
        base["avg_hold_minutes"] = round(sum(hold_vals) / len(hold_vals), 2) if hold_vals else 0.0

        result[phase] = base

    # Summary keys — only consider phases with trades
    phases_with_trades = {p: result[p] for p in [*VALID_SESSION_PHASES, "unknown"] if result[p]["n_trades"] > 0}

    if phases_with_trades:
        result["best_session"] = max(phases_with_trades, key=lambda p: phases_with_trades[p]["avg_pnl"])
        result["worst_session"] = min(phases_with_trades, key=lambda p: phases_with_trades[p]["avg_pnl"])
        result["recommended_sessions"] = sorted(p for p, s in phases_with_trades.items() if s["profit_factor"] > 1.2)
        result["avoid_sessions"] = sorted(p for p, s in phases_with_trades.items() if 0 < s["profit_factor"] < 0.8)
    else:
        result["best_session"] = None
        result["worst_session"] = None
        result["recommended_sessions"] = []
        result["avoid_sessions"] = []

    return result


def compute_enrichment_breakdown(
    trades: list[dict],
    pnl_key: str = "pnl",
) -> dict[str, Any]:
    """Break down performance by enrichment labels.

    Groups:
    - near_earnings True vs False
    - near_fomc True vs False
    - opex_week True vs False
    - By session_phase
    - By liquidity regime
    - By correlation regime

    Returns a dict keyed by enrichment category, each containing sub-group stats.
    Missing fields are skipped gracefully.
    """
    result: dict[str, Any] = {}

    # Boolean enrichment fields
    for field in ENRICHMENT_BOOL_FIELDS:
        true_pnls: list[float] = []
        false_pnls: list[float] = []
        found = False
        for t in trades:
            val = t.get(field)
            if val is None:
                continue
            found = True
            pnl = float(t.get(pnl_key, 0.0))
            if val:
                true_pnls.append(pnl)
            else:
                false_pnls.append(pnl)

        if found:
            label = field.replace("entry_", "")
            result[label] = {
                "True": _group_stats(true_pnls),
                "False": _group_stats(false_pnls),
            }

    # Categorical enrichment fields
    categorical_fields = [
        (SESSION_PHASE_FIELD, "session_phase"),
        (REGIME_LIQUIDITY_FIELD, "liquidity"),
        (REGIME_CORRELATION_FIELD, "correlation"),
    ]
    for field, label in categorical_fields:
        breakdown = compute_regime_breakdown(trades, regime_field=field, pnl_key=pnl_key)
        if breakdown:
            result[label] = breakdown

    return result


# ---------------------------------------------------------------------------
# Exit regime fields (Phase 1.2)
# ---------------------------------------------------------------------------

EXIT_REGIME_TREND_FIELD = "exit_regime_trend"
EXIT_REGIME_VOL_FIELD = "exit_regime_vol"

# Volatility levels considered "expanded"
_VOL_LOW = {"LOW", "NORMAL"}
_VOL_HIGH = {"HIGH", "SHOCK"}

# Liquidity levels considered "drained"
_LIQ_GOOD = {"LIQUID", "NORMAL", "GOOD"}
_LIQ_THIN = {"THIN", "DRY", "STRESSED"}


def compute_regime_transitions(
    trades: list[dict],
    pnl_key: str = "pnl",
) -> dict[str, Any]:
    """Analyze regime changes between entry and exit for each trade.

    Returns a dict with:
    - total_trades: number of trades with both entry and exit regime fields
    - same_regime: count where entry == exit on both trend and vol
    - regime_changed: count where at least one axis changed
    - transition_matrix: keyed by "ENTRY_TREND->EXIT_TREND" with count/avg_pnl/win_rate
    - adverse_transitions: list of detected adverse transition patterns
    - pnl_by_stability: "stable" vs "changed" PnL comparison
    """
    transition_pnls: dict[str, list[float]] = defaultdict(list)
    stable_pnls: list[float] = []
    changed_pnls: list[float] = []
    total = 0

    # Adverse transition accumulators
    adverse: dict[str, list[float]] = defaultdict(list)

    for t in trades:
        entry_trend = t.get(REGIME_TREND_FIELD)
        exit_trend = t.get(EXIT_REGIME_TREND_FIELD)
        entry_vol = t.get(REGIME_VOL_FIELD)
        exit_vol = t.get(EXIT_REGIME_VOL_FIELD)

        # Skip trades missing any of the four regime fields
        if entry_trend is None or exit_trend is None or entry_vol is None or exit_vol is None:
            continue

        entry_trend = str(entry_trend)
        exit_trend = str(exit_trend)
        entry_vol = str(entry_vol)
        exit_vol = str(exit_vol)
        pnl = float(t.get(pnl_key, 0.0))
        side = str(t.get("side", "")).lower()
        total += 1

        # Trend transition matrix
        trend_key = f"{entry_trend}->{exit_trend}"
        transition_pnls[trend_key].append(pnl)

        # Stability classification (both axes must match)
        is_stable = entry_trend == exit_trend and entry_vol == exit_vol
        if is_stable:
            stable_pnls.append(pnl)
        else:
            changed_pnls.append(pnl)

        # --- Adverse transition detection ---

        # Vol expansion (entry LOW/NORMAL -> exit HIGH/SHOCK)
        if entry_vol.upper() in _VOL_LOW and exit_vol.upper() in _VOL_HIGH:
            if side == "sell":
                adverse["vol_expansion_short"].append(pnl)
            elif side == "buy":
                adverse["vol_expansion_long"].append(pnl)

        # Trend reversal (UP->DOWN or DOWN->UP)
        if (entry_trend.upper() == "UP" and exit_trend.upper() == "DOWN") or (
            entry_trend.upper() == "DOWN" and exit_trend.upper() == "UP"
        ):
            adverse["trend_reversal"].append(pnl)

        # Liquidity drain
        entry_liq = str(t.get(REGIME_LIQUIDITY_FIELD, "")).upper()
        exit_liq = str(t.get("exit_liquidity", "")).upper()
        if entry_liq in _LIQ_GOOD and exit_liq in _LIQ_THIN:
            adverse["liquidity_drain"].append(pnl)

    if total == 0:
        return {}

    # Build transition matrix
    transition_matrix: dict[str, dict[str, Any]] = {}
    for key, pnls in sorted(transition_pnls.items()):
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        transition_matrix[key] = {
            "count": n,
            "avg_pnl": round(sum(pnls) / n, 4) if n > 0 else 0.0,
            "win_rate": round(wins / n, 4) if n > 0 else 0.0,
        }

    # Build adverse transitions list
    adverse_transitions: list[dict[str, Any]] = []
    for atype, pnls in sorted(adverse.items()):
        n = len(pnls)
        adverse_transitions.append(
            {
                "type": atype,
                "count": n,
                "avg_pnl": round(sum(pnls) / n, 4) if n > 0 else 0.0,
            }
        )

    # PnL by stability
    def _stability_stats(pnls: list[float]) -> dict[str, Any]:
        n = len(pnls)
        if n == 0:
            return {"n": 0, "avg_pnl": 0.0}
        return {"n": n, "avg_pnl": round(sum(pnls) / n, 4)}

    return {
        "total_trades": total,
        "same_regime": len(stable_pnls),
        "regime_changed": len(changed_pnls),
        "transition_matrix": transition_matrix,
        "adverse_transitions": adverse_transitions,
        "pnl_by_stability": {
            "stable": _stability_stats(stable_pnls),
            "changed": _stability_stats(changed_pnls),
        },
    }


def compute_event_attribution(
    trades: list[dict],
    pnl_key: str = "pnl",
) -> dict[str, Any]:
    """Detailed P&L attribution around market events.

    Splits trades by proximity to earnings, FOMC, and OPEX events, then
    computes performance statistics for each group.  Also detects trades
    that coincide with multiple events simultaneously.

    Returns::

        {
            "earnings": {
                "trades_near": N, "trades_far": N,
                "pnl_near": X, "pnl_far": X,
                "winrate_near": Y, "winrate_far": Y,
                "avg_pnl_near": X, "avg_pnl_far": X,
                "near_is_better": bool,
                "recommendation": "TRADE" | "AVOID" | "NEUTRAL"
            },
            "fomc": { ... same structure ... },
            "opex": { ... same structure ... },
            "combined_events": {
                "multi_event_trades": N,
                "multi_event_pnl": X,
            }
        }

    Recommendation logic:

    - AVOID if near-event winrate < 40% AND avg_pnl_near < 0
    - TRADE if near-event winrate > 55% AND avg_pnl_near > avg_pnl_far
    - NEUTRAL otherwise
    """
    event_fields = {
        "earnings": "entry_near_earnings",
        "fomc": "entry_near_fomc",
        "opex": "entry_opex_week",
    }

    result: dict[str, Any] = {}

    for event_name, field in event_fields.items():
        near_pnls: list[float] = []
        far_pnls: list[float] = []
        found = False

        for t in trades:
            val = t.get(field)
            if val is None:
                continue
            found = True
            pnl = float(t.get(pnl_key, 0.0))
            if val:
                near_pnls.append(pnl)
            else:
                far_pnls.append(pnl)

        if not found:
            continue

        n_near = len(near_pnls)
        n_far = len(far_pnls)
        pnl_near = sum(near_pnls)
        pnl_far = sum(far_pnls)
        winrate_near = _safe_div(sum(1 for p in near_pnls if p > 0), n_near)
        winrate_far = _safe_div(sum(1 for p in far_pnls if p > 0), n_far)
        avg_pnl_near = _safe_div(pnl_near, n_near)
        avg_pnl_far = _safe_div(pnl_far, n_far)

        near_is_better = avg_pnl_near > avg_pnl_far

        # Recommendation logic
        if winrate_near < 0.40 and avg_pnl_near < 0:
            recommendation = "AVOID"
        elif winrate_near > 0.55 and avg_pnl_near > avg_pnl_far:
            recommendation = "TRADE"
        else:
            recommendation = "NEUTRAL"

        result[event_name] = {
            "trades_near": n_near,
            "trades_far": n_far,
            "pnl_near": round(pnl_near, 4),
            "pnl_far": round(pnl_far, 4),
            "winrate_near": round(winrate_near, 4),
            "winrate_far": round(winrate_far, 4),
            "avg_pnl_near": round(avg_pnl_near, 4),
            "avg_pnl_far": round(avg_pnl_far, 4),
            "near_is_better": near_is_better,
            "recommendation": recommendation,
        }

    # Combined events: trades near multiple events simultaneously
    multi_event_pnls: list[float] = []
    for t in trades:
        flags = [t.get(f) for f in event_fields.values()]
        # Skip trades with no event flags at all
        if all(f is None for f in flags):
            continue
        true_count = sum(1 for f in flags if f is True)
        if true_count >= 2:
            multi_event_pnls.append(float(t.get(pnl_key, 0.0)))

    result["combined_events"] = {
        "multi_event_trades": len(multi_event_pnls),
        "multi_event_pnl": round(sum(multi_event_pnls), 4),
    }

    return result


def format_regime_report(breakdowns: dict[str, Any]) -> str:
    """Format regime breakdowns as a readable string for logging/display."""
    if not breakdowns:
        return "No regime data available."

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("REGIME PERFORMANCE BREAKDOWN")
    lines.append("=" * 70)

    for section_name, section_data in breakdowns.items():
        lines.append("")
        lines.append(f"--- {section_name} ---")

        if isinstance(section_data, dict) and "cells" in section_data:
            # Matrix format
            rows = section_data.get("rows", [])
            cols = section_data.get("cols", [])
            cells = section_data.get("cells", {})

            header = f"{'':>12}" + "".join(f"{c:>14}" for c in cols)
            lines.append(header)

            for r in rows:
                row_parts = [f"{r:>12}"]
                for c in cols:
                    key = f"{r}|{c}"
                    cell = cells.get(key, {})
                    n = cell.get("n_trades", 0)
                    wr = cell.get("win_rate", 0.0)
                    row_parts.append(f"{n:>4}t {wr:>5.1%}" if n > 0 else f"{'--':>14}")
                lines.append("".join(row_parts))
        elif isinstance(section_data, dict):
            # Group stats format
            for group_name, group_stats in section_data.items():
                if not isinstance(group_stats, dict):
                    continue
                n = group_stats.get("n_trades", 0)
                wr = group_stats.get("win_rate", 0.0)
                pf = group_stats.get("profit_factor", 0.0)
                avg = group_stats.get("avg_pnl", 0.0)
                total = group_stats.get("total_pnl", 0.0)
                lines.append(
                    f"  {group_name:>12}: {n:>4} trades | WR {wr:>5.1%} | PF {pf:>6.2f} | "
                    f"avg {avg:>8.2f} | total {total:>10.2f}"
                )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
