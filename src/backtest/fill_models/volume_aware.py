from __future__ import annotations

from typing import TYPE_CHECKING

from src.backtest.fill_models.protocol import FillResult

if TYPE_CHECKING:
    from src.core.domain import Bar


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

    def compute_fill(
        self,
        order_side: str,
        order_qty: int,
        order_price: float | None,
        order_type: str,
        bar: Bar | None,
    ) -> FillResult:
        price = order_price or 0.0
        bar_volume = getattr(bar, "volume", 0) if bar else 0

        if bar_volume > 0:
            participation = order_qty / bar_volume
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

        commission = max(self.min_commission, self.commission_per_share * order_qty)
        return FillResult(
            fill_price=fill_price,
            filled_qty=order_qty,
            commission=commission,
            slippage_bps=effective_bps,
            market_impact=impact_bps,
        )
