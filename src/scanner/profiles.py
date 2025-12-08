from abc import ABC, abstractmethod
from typing import Dict, Any
from src.data.models import SymbolFeatures

class ScannerProfile(ABC):
    """
    Base class for strategy-specific scanner profiles.
    """
    @abstractmethod
    def filter(self, features: SymbolFeatures) -> bool:
        """
        Returns True if the symbol passes the filter for this strategy.
        """
        pass

class VWAPReversionProfile(ScannerProfile):
    """
    Scanner profile for VWAP Reversion strategy.
    """
    def __init__(self, min_price: float = 10.0, min_volume: float = 100000):
        self.min_price = min_price
        self.min_volume = min_volume

    def filter(self, features: SymbolFeatures) -> bool:
        # Basic liquidity checks
        if features.price < self.min_price:
            return False
        if features.volume < self.min_volume:
            return False
            
        # Strategy specific checks (e.g. high volatility, or specific flow)
        # For now, just liquidity.
        return True
