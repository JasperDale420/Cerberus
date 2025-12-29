from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.cooldown_bars = int(config.get("cooldown_bars", 5))
        from datetime import datetime

        self.last_signal_time: Dict[str, datetime] = {}

    def _check_cooldown(self, symbol: str, current_time: Any) -> bool:
        """
        Returns True if cooldown has passed and it's safe to signal.
        """
        if self.cooldown_bars <= 0:
            return True
        last = self.last_signal_time.get(symbol)
        if last is None:
            return True
        # Assume 1 minute per bar for now (safe default for scalping)
        from datetime import timedelta

        delta = timedelta(minutes=self.cooldown_bars)
        if current_time - last < delta:
            return False
        return True

    def _create_signal(
        self,
        symbol: str,
        side: OrderSide,
        bar: Bar,
        market_state: MarketState,
        stop_price: float,
        target_price: float,
        entry_price: Optional[float] = None,
        size_hint: float = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """
        Create a Signal with standardized parameters.

        This helper eliminates duplicate Signal construction code across strategies.

        Args:
            symbol: Symbol to trade
            side: OrderSide.BUY or OrderSide.SELL
            bar: Current bar being processed
            market_state: Current market state
            stop_price: Stop loss price
            target_price: Take profit price
            entry_price: Entry price (defaults to bar.close)
            size_hint: Size hint for position sizing (default 0)
            meta: Optional metadata dictionary

        Returns:
            Signal object ready to be processed by execution engine
        """
        return Signal(
            symbol=symbol,
            side=side,
            size_hint=size_hint,
            entry_price=entry_price if entry_price is not None else bar.close,
            stop_price=stop_price,
            target_price=target_price,
            strategy=self.name,
            regime=market_state.regime,
            generated_at=bar.time,
            meta=meta or {},
        )

    @abstractmethod
    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """
        Process a new bar and potentially return a Signal.
        """
        pass
