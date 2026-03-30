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
