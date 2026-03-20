from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core.domain import Bar


@runtime_checkable
class FillModel(Protocol):
    """Protocol for pluggable fill simulation models."""

    def compute_fill(
        self,
        order_side: str,
        order_qty: int,
        order_price: float | None,
        order_type: str,
        bar: Bar | None,
    ) -> FillResult: ...


@dataclass(frozen=True, slots=True)
class FillResult:
    fill_price: float
    filled_qty: int
    commission: float
    slippage_bps: float
    market_impact: float
