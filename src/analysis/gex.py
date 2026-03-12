"""GEX (Gamma Exposure) analyzer for dealer positioning regime classification.

Computes net gamma exposure from options chain data and classifies the
dealer hedging regime: positive GEX = mean-reverting, negative = trending.

Papers: Barbon & Buraschi (2021) — "Gamma Fragility"
        Koychev (2024) — "GEX and Realized Volatility"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.core.logger import StructuredLogger


@dataclass
class GEXResult:
    """Result of a GEX computation."""

    net_gex: float  # Total net gamma exposure in dollar terms
    gex_regime: str  # "positive", "negative", "neutral"
    gamma_flip_distance: float  # % distance from current price to gamma flip level
    put_call_gex_ratio: float  # put GEX / call GEX
    top_strikes: list[tuple[float, float]] = field(default_factory=list)  # (strike, gex_value)


class GEXAnalyzer:
    """Analyzes options chain gamma exposure to classify dealer hedging regime.

    When dealers are long gamma (positive GEX), they hedge by buying dips and
    selling rips, creating mean-reverting dynamics. When short gamma (negative GEX),
    dealer hedging amplifies moves, creating trending dynamics.

    Args:
        logger: Optional structured logger instance.
        neutral_band_pct: Fraction of total absolute GEX below which the regime
            is classified as "neutral". Defaults to 0.05 (5%).
    """

    def __init__(
        self,
        logger: Optional[StructuredLogger] = None,
        neutral_band_pct: float = 0.05,
    ) -> None:
        self._logger = logger
        self._neutral_band_pct = neutral_band_pct
        self._history: deque[GEXResult] = deque(maxlen=20)
        self._current_regime: Optional[str] = None

    def update(self, current_price: float, chain_data: list[dict]) -> Optional[GEXResult]:
        """Compute GEX from raw options chain data.

        Args:
            current_price: Current underlying price (S).
            chain_data: List of dicts with keys: strike, call_oi, put_oi,
                call_gamma, put_gamma.

        Returns:
            GEXResult or None if chain_data is empty or current_price <= 0.
        """
        if not chain_data or current_price <= 0:
            return None

        s_sq = current_price * current_price
        strike_gex: list[tuple[float, float]] = []
        total_call_gex = 0.0
        total_put_gex = 0.0

        for row in chain_data:
            strike = float(row.get("strike", 0))
            call_oi = float(row.get("call_oi", 0))
            put_oi = float(row.get("put_oi", 0))
            call_gamma = float(row.get("call_gamma", 0))
            put_gamma = float(row.get("put_gamma", 0))

            # GEX per strike: OI * gamma * 100 shares * S² * 0.01
            call_gex = call_oi * call_gamma * 100.0 * s_sq * 0.01
            put_gex = put_oi * put_gamma * 100.0 * s_sq * 0.01

            # Net GEX at this strike: calls contribute positive, puts contribute negative
            net_at_strike = call_gex - put_gex
            strike_gex.append((strike, net_at_strike))
            total_call_gex += call_gex
            total_put_gex += put_gex

        net_gex = total_call_gex - total_put_gex

        # Gamma flip level: strike where cumulative GEX changes sign
        gamma_flip_price = self._find_gamma_flip(strike_gex)
        if gamma_flip_price is not None:
            gamma_flip_distance = (gamma_flip_price - current_price) / current_price
        else:
            gamma_flip_distance = 0.0

        # Put/call GEX ratio
        put_call_gex_ratio = total_put_gex / total_call_gex if total_call_gex > 1e-12 else 0.0

        # Top strikes by absolute GEX
        sorted_strikes = sorted(strike_gex, key=lambda x: abs(x[1]), reverse=True)
        top_strikes = sorted_strikes[:5]

        # Classify regime
        total_abs_gex = abs(total_call_gex) + abs(total_put_gex)
        neutral_band = total_abs_gex * self._neutral_band_pct

        if net_gex > neutral_band:
            regime = "positive"
        elif net_gex < -neutral_band:
            regime = "negative"
        else:
            regime = "neutral"

        result = GEXResult(
            net_gex=net_gex,
            gex_regime=regime,
            gamma_flip_distance=gamma_flip_distance,
            put_call_gex_ratio=put_call_gex_ratio,
            top_strikes=top_strikes,
        )

        self._history.append(result)
        self._current_regime = regime

        if self._logger:
            self._logger.debug(
                "GEX updated",
                net_gex=net_gex,
                regime=regime,
                gamma_flip_distance=gamma_flip_distance,
                put_call_gex_ratio=put_call_gex_ratio,
            )

        return result

    def update_from_normalized(self, current_price: float, gex_data: dict) -> Optional[GEXResult]:
        """Compute GEX from pre-aggregated Data-Gateway NormalizedGreekExposure format.

        This path is used when consuming UW GEX API data via Data-Gateway, where
        gamma values are already aggregated across the chain.

        Args:
            current_price: Current underlying price.
            gex_data: Dict with keys: call_gamma, put_gamma (already aggregated).
                Optionally: call_delta, put_delta (unused for GEX but logged).

        Returns:
            GEXResult or None if data is insufficient.
        """
        if not gex_data or current_price <= 0:
            return None

        call_gamma = float(gex_data.get("call_gamma", 0))
        put_gamma = float(gex_data.get("put_gamma", 0))

        if call_gamma == 0 and put_gamma == 0:
            return None

        s_sq = current_price * current_price

        # Pre-aggregated: treat as total gamma across all strikes
        total_call_gex = call_gamma * 100.0 * s_sq * 0.01
        total_put_gex = put_gamma * 100.0 * s_sq * 0.01

        net_gex = total_call_gex - total_put_gex
        put_call_gex_ratio = total_put_gex / total_call_gex if total_call_gex > 1e-12 else 0.0

        # No per-strike data available, so no gamma flip or top strikes
        total_abs_gex = abs(total_call_gex) + abs(total_put_gex)
        neutral_band = total_abs_gex * self._neutral_band_pct

        if net_gex > neutral_band:
            regime = "positive"
        elif net_gex < -neutral_band:
            regime = "negative"
        else:
            regime = "neutral"

        result = GEXResult(
            net_gex=net_gex,
            gex_regime=regime,
            gamma_flip_distance=0.0,
            put_call_gex_ratio=put_call_gex_ratio,
            top_strikes=[],
        )

        self._history.append(result)
        self._current_regime = regime

        if self._logger:
            self._logger.debug(
                "GEX updated (normalized)",
                net_gex=net_gex,
                regime=regime,
            )

        return result

    def get_regime(self) -> Optional[str]:
        """Return the current GEX regime string, or None if never computed."""
        return self._current_regime

    def _find_gamma_flip(self, strike_gex: list[tuple[float, float]]) -> Optional[float]:
        """Find the price level where cumulative GEX changes sign.

        Walks strikes from lowest to highest, accumulating GEX. The flip level
        is where the cumulative sum crosses zero.

        Returns:
            The interpolated strike price at the flip, or None if no sign change.
        """
        if not strike_gex:
            return None

        sorted_strikes = sorted(strike_gex, key=lambda x: x[0])
        cumulative = 0.0
        prev_cum = 0.0
        prev_strike = sorted_strikes[0][0]

        for i, (strike, gex_val) in enumerate(sorted_strikes):
            cumulative += gex_val
            if i > 0 and prev_cum * cumulative < 0:
                # Sign change: interpolate between prev_strike and strike
                if abs(cumulative - prev_cum) > 1e-12:
                    frac = abs(prev_cum) / abs(cumulative - prev_cum)
                    return prev_strike + frac * (strike - prev_strike)
                return strike
            prev_cum = cumulative
            prev_strike = strike

        return None
