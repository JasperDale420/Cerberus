from __future__ import annotations

from typing import TYPE_CHECKING

from src.backtest.fill_models.protocol import FillResult

if TYPE_CHECKING:
    from src.core.domain import Bar


class FixedSlippageFillModel:
    """Original fixed-BPS slippage model. Extracted from SimulatedOrderExecutor."""

    def __init__(self, slippage_bps: float = 2.0, commission_per_share: float = 0.001, min_commission: float = 0.0):
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

    def compute_fill(
        self,
        order_side: str,
        order_qty: int,
        order_price: float | None,
        order_type: str,
        bar: Bar | None,
    ) -> FillResult:
        price = order_price or 0.0
        slip_frac = self.slippage_bps / 10_000.0
        if order_side == "buy":
            fill_price = price * (1.0 + slip_frac)
        else:
            fill_price = price * (1.0 - slip_frac)
        commission = max(self.min_commission, self.commission_per_share * order_qty)
        return FillResult(
            fill_price=fill_price,
            filled_qty=order_qty,
            commission=commission,
            slippage_bps=self.slippage_bps,
            market_impact=0.0,
        )
