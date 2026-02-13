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

            def _is_finite_number(value: object) -> bool:
                return isinstance(value, (int, float)) and math.isfinite(value)

            # Basic sanity checks
            if not features:
                return False

            # Price validity
            if not _is_finite_number(features.price):
                if self.logger:
                    self.logger.warning(
                        "Non-finite price",
                        symbol=getattr(features, "symbol", "UNKNOWN"),
                        price=getattr(features, "price", None),
                    )
                return False
            if features.price <= 0:
                if self.logger:
                    self.logger.warning("Invalid price", symbol=features.symbol, price=features.price)
                return False

            # Filter criteria
            if features.price < min_price or features.price > max_price:
                return False

            if not _is_finite_number(features.avg_volume):
                if self.logger:
                    self.logger.warning(
                        "Non-finite average volume",
                        symbol=getattr(features, "symbol", "UNKNOWN"),
                        avg_volume=getattr(features, "avg_volume", None),
                    )
                return False
            if features.avg_volume < min_volume:
                return False

            # Volatility checks (ATR)
            if not _is_finite_number(features.atr_pct):
                if self.logger:
                    self.logger.warning(
                        "Non-finite ATR percent",
                        symbol=getattr(features, "symbol", "UNKNOWN"),
                        atr_pct=getattr(features, "atr_pct", None),
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
                    exc_info=True,
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
                if not math.isfinite(features.flow_zscore):
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
                    if val is not None and isinstance(val, (int, float)):
                        if not math.isfinite(val):
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
                    exc_info=True,
                )
            return False
