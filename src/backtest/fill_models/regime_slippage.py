"""Regime-dependent slippage multipliers for backtest fill models.

Applies volatility, liquidity, and event-driven multipliers to base slippage
so that fills are more realistic across different market conditions.
"""

from __future__ import annotations

from typing import Any, Dict

# Volatility regime multipliers
_VOL_MULT: Dict[str, float] = {
    "low": 0.5,
    "normal": 1.0,
    "high": 2.0,
    "shock": 4.0,
}

# Liquidity regime multipliers
_LIQ_MULT: Dict[str, float] = {
    "liquid": 0.7,
    "good": 0.7,
    "normal": 1.0,
    "thin": 2.0,
    "dry": 5.0,
    "stressed": 3.0,
}

# Event multipliers (take the max of all active events)
_EVENT_MULT: Dict[str, float] = {
    "opex_week": 1.2,
    "near_earnings": 1.3,
    "near_fomc": 1.2,
}


def compute_regime_slippage_multiplier(regime_labels: Dict[str, Any]) -> float:
    """Compute a combined slippage multiplier from regime labels.

    Args:
        regime_labels: Dict with keys like 'regime_vol', 'liquidity_regime',
            'earnings_window', etc. Values are strings (e.g. 'HIGH', 'THIN').

    Returns:
        Combined multiplier >= 0.1. Falls back to 1.0 for missing/unknown labels.
    """
    if not regime_labels:
        return 1.0

    # --- Volatility axis ---
    vol_label = str(regime_labels.get("regime_vol", "normal")).lower()
    vol_mult = _VOL_MULT.get(vol_label, 1.0)

    # --- Liquidity axis ---
    liq_label = str(regime_labels.get("liquidity_regime", "normal")).lower()
    liq_mult = _LIQ_MULT.get(liq_label, 1.0)

    # --- Event axis (take max of all active events) ---
    event_mults: list[float] = []

    earnings = regime_labels.get("earnings_window")
    if earnings and str(earnings).lower() not in ("none", "false", "0", ""):
        event_mults.append(_EVENT_MULT["near_earnings"])

    # Future-proof: support opex_week and near_fomc if/when added to regime labels
    if regime_labels.get("opex_week"):
        event_mults.append(_EVENT_MULT["opex_week"])
    if regime_labels.get("near_fomc"):
        event_mults.append(_EVENT_MULT["near_fomc"])

    event_mult = max(event_mults) if event_mults else 1.0

    combined = vol_mult * liq_mult * event_mult
    return max(combined, 0.1)  # Floor to prevent zero/negative
