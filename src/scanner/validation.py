import math
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

    @staticmethod
    def _is_finite(value: object) -> bool:
        try:
            return math.isfinite(value)
        except Exception:
            return False

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
            if not self._is_finite(features.price):
                if self.logger:
                    self.logger.warning("Invalid price", symbol=features.symbol, price=features.price)
                return False
            if features.price <= 0:
                if self.logger:
                    self.logger.warning("Invalid price", symbol=features.symbol, price=features.price)
                return False

            # Filter criteria
            if features.price < min_price or features.price > max_price:
                return False

            if not self._is_finite(features.avg_volume):
                if self.logger:
                    self.logger.warning(
                        "Invalid volume",
                        symbol=features.symbol,
                        avg_volume=features.avg_volume,
                    )
                return False
            if features.avg_volume < min_volume:
                return False

            # Volatility checks (ATR)
            if not self._is_finite(features.atr_pct):
                if self.logger:
                    self.logger.warning(
                        "Invalid atr_pct",
                        symbol=features.symbol,
                        atr_pct=features.atr_pct,
                    )
                return False
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
        Validates flow metrics for NaN, Inf, and basic sanity.

        P1 fix: Properly check for non-finite values that could corrupt scoring.
        """
        try:
            # Check flow_zscore for NaN/Inf
            if features.flow_zscore is not None:
                if not self._is_finite(features.flow_zscore):
                    if self.logger:
                        self.logger.warning(
                            "Invalid flow_zscore",
                            symbol=features.symbol,
                            flow_zscore=features.flow_zscore,
                        )
                    return False

            # Check extra flow metrics if present
            if features.extra:
                raw_count = features.extra.get("flow_raw_count", 0)
                if raw_count is not None and raw_count < 0:
                    return False

                # Check for any NaN/Inf in extra numeric fields
                for key in ("flow_premium", "flow_volume", "call_volume", "put_volume"):
                    val = features.extra.get(key)
                    if val is not None and not self._is_finite(val):
                        if self.logger:
                            self.logger.warning(
                                "Invalid flow metric",
                                symbol=features.symbol,
                                key=key,
                                value=val,
                            )
                        return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Flow validation failed",
                    symbol=getattr(features, "symbol", "UNKNOWN"),
                    error=str(e),
                )
            return False
