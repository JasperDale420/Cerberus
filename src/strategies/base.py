from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, Signal, SymbolState
from src.core.logger import StructuredLogger


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        self.config = config
        self.logger = logger

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
