from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.cooldown_bars = int(config.get("cooldown_bars", 5))
        # M5 fix: Configurable bar duration for accurate cooldown across timeframes
        self.bar_duration_minutes = float(config.get("bar_duration_minutes", 1.0))
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
        # M5 fix: Use configured bar duration instead of hardcoded 1 minute
        delta = timedelta(minutes=self.cooldown_bars * self.bar_duration_minutes)
        if current_time - last < delta:
            return False
        return True

    def _require_min_bars(
        self, symbol_state: SymbolState, min_count: int, log: bool = True
    ) -> bool:
        """
        Check if symbol_state has minimum required bars for analysis.

        This helper eliminates duplicate bar count validation across strategies.

        Args:
            symbol_state: Symbol state containing bars
            min_count: Minimum number of bars required
            log: Whether to log when insufficient bars (default True)

        Returns:
            True if sufficient bars available, False otherwise

        Example:
            if not self._require_min_bars(symbol_state, 20):
                return None
        """
        bars = symbol_state.bars
        if not bars or len(bars) < min_count:
            if log:
                self.logger.debug(
                    f"{self.name}: insufficient bars",
                    min_required=min_count,
                    available=len(bars) if bars else 0,
                )
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
        # Extract multi-axis regime data from snapshot
        snapshot = market_state.regime_snapshot
        regime_tags = snapshot.regime_tags if snapshot else {}
        regime_confidence = snapshot.confidence if snapshot else {}

        return Signal(
            symbol=symbol,
            side=side,
            size_hint=size_hint,
            entry_price=entry_price if entry_price is not None else bar.close,
            stop_price=stop_price,
            target_price=target_price,
            strategy=self.name,
            generated_at=bar.time,
            meta=meta or {},
            regime_tags=regime_tags,
            regime_confidence=regime_confidence,
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
