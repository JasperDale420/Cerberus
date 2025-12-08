from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from src.analysis.regime import Regime
from src.core.logger import StructuredLogger
from src.data.models import Bar

@dataclass
class Signal:
    symbol: str
    side: str  # "buy" or "sell"
    size_hint: float
    entry_price: float
    stop_price: float
    target_price: float
    strategy: str
    regime: Regime
    generated_at: datetime
    meta: Dict[str, Any]

@dataclass
class SymbolState:
    symbol: str
    bars: List[Bar] 
    position: Optional[Any] # Position
    # Add other fields as needed

@dataclass
class MarketState:
    time: datetime
    regime: Regime
    # Add other fields as needed

class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def on_bar(self,
               symbol: str,
               bar: Bar,
               symbol_state: SymbolState,
               market_state: MarketState) -> Optional[Signal]:
        """
        Process a new bar and potentially return a Signal.
        """
        pass
