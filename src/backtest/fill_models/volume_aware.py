from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.backtest.fill_models.protocol import FillResult

if TYPE_CHECKING:
    from src.core.domain import Bar

# Participation-rate caps by liquidity regime.
# Each value is the maximum fraction of bar volume that may be filled.
LIQUIDITY_PARTICIPATION_CAPS: dict[str, float] = {
    "LIQUID": 0.25,
    "NORMAL": 0.15,
    "THIN": 0.05,
    "DRY": 0.02,
}

# Minimum dollar volume (close * volume) required for any fill.
MIN_DOLLAR_VOLUME = 1_000_000.0


@dataclass
class FillStats:
    """Counters for fill-quality diagnostics."""

    total_orders: int = 0
    full_fills: int = 0
    partial_fills: int = 0
    rejected_fills: int = 0


class VolumeAwareFillModel:
    """Slippage scales with order participation rate in bar volume."""

    def __init__(
        self,
        base_slippage_bps: float = 2.0,
        impact_coefficient: float = 200.0,
        commission_per_share: float = 0.001,
        min_commission: float = 0.0,
        max_slippage_bps: float = 50.0,
    ):
        self.base_slippage_bps = base_slippage_bps
        self.impact_coefficient = impact_coefficient
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.max_slippage_bps = max_slippage_bps
        self._stats = FillStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_fill(
        self,
        order_side: str,
        order_qty: int,
        order_price: float | None,
        order_type: str,
        bar: Bar | None,
        regime_liquidity: str | None = None,
    ) -> FillResult:
        self._stats.total_orders += 1
        price = order_price or 0.0
        bar_volume = getattr(bar, "volume", 0) if bar else 0
        bar_close = getattr(bar, "close", price) if bar else price

        # --- Minimum dollar-volume gate (active when regime_liquidity is provided) ---
        if regime_liquidity is not None:
            dollar_volume = bar_close * bar_volume if bar_volume > 0 else 0.0
            if dollar_volume < MIN_DOLLAR_VOLUME:
                self._stats.rejected_fills += 1
                return FillResult(
                    fill_price=0.0,
                    filled_qty=0,
                    commission=0.0,
                    slippage_bps=0.0,
                    market_impact=0.0,
                )

        # --- Liquidity-regime participation cap ---
        fill_qty = order_qty
        if regime_liquidity is not None and bar_volume > 0:
            cap_pct = LIQUIDITY_PARTICIPATION_CAPS.get(regime_liquidity.upper(), 0.15)
            max_shares = int(bar_volume * cap_pct)
            if max_shares < 1:
                max_shares = 1
            fill_qty = min(order_qty, max_shares)

        if fill_qty <= 0:
            self._stats.rejected_fills += 1
            return FillResult(
                fill_price=0.0,
                filled_qty=0,
                commission=0.0,
                slippage_bps=0.0,
                market_impact=0.0,
            )

        # --- Slippage calculation (uses fill_qty, not order_qty) ---
        if bar_volume > 0:
            participation = fill_qty / bar_volume
            impact_bps = participation * self.impact_coefficient
            effective_bps = min(self.base_slippage_bps + impact_bps, self.max_slippage_bps)
        else:
            effective_bps = self.max_slippage_bps
            impact_bps = effective_bps - self.base_slippage_bps

        slip_frac = effective_bps / 10_000.0
        if order_side == "buy":
            fill_price = price * (1.0 + slip_frac)
        else:
            fill_price = price * (1.0 - slip_frac)

        commission = max(self.min_commission, self.commission_per_share * fill_qty)

        # --- Stats tracking ---
        if fill_qty < order_qty:
            self._stats.partial_fills += 1
        else:
            self._stats.full_fills += 1

        return FillResult(
            fill_price=fill_price,
            filled_qty=fill_qty,
            commission=commission,
            slippage_bps=effective_bps,
            market_impact=impact_bps,
        )

    def get_fill_stats(self) -> FillStats:
        """Return accumulated fill statistics."""
        return self._stats
