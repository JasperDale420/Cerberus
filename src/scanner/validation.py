from typing import Optional

from src.core.domain import SymbolFeatures
from src.core.logger import StructuredLogger


class DataValidator:
    """
    Validates market data quality for the Scanner.
    Functions as a Quality Gate to prevent processing of stale or invalid data.
    """

    def __init__(self, logger: Optional[StructuredLogger] = None):
        self.logger = logger

    def validate_technicals(
        self,
        features: SymbolFeatures,
        min_price: float = 0.0,
        max_price: float = float("inf"),
        min_volume: float = 0.0,
        min_atr_pct: float = 0.0,
        max_atr_pct: float = float("inf"),
    ) -> bool:
        """
        Validates technical indicators and basic price/volume data.
        """
        try:
            # Basic sanity checks
            if not features:
                return False

            # Price validity
            if features.price <= 0:
                if self.logger:
                    self.logger.warning(
                        "Invalid price", symbol=features.symbol, price=features.price
                    )
                return False

            # Filter criteria
            if features.price < min_price or features.price > max_price:
                return False

            if features.avg_volume < min_volume:
                return False

            # Volatility checks (ATR)
            if features.atr_pct < min_atr_pct or features.atr_pct > max_atr_pct:
                return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Technical validation failed",
                    symbol=getattr(features, "symbol", "UNKNOWN"),
                    error=str(e),
                )
            return False

    def validate_flow(self, features: SymbolFeatures) -> bool:
        """
        Validates flow metrics.
        """
        # Currently just a placeholder for future flow sanity checks
        # e.g., if flow_zscore is infinite or NaN
        if features.extra and features.extra.get("flow_raw_count", 0) < 0:
            return False
        return True
