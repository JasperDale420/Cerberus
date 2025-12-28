from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, Signal, SymbolState
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
