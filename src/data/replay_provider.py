"""
Replay Provider: Provides historical data from snapshots for offline backtesting.
Drop-in replacement for live Fetcher when running backtests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import ExternalSnapshot, FeatureSnapshot
from src.core.domain import SymbolFeatures
from src.core.logger import StructuredLogger


class ReplayProvider:
    """
    Provides data from persisted snapshots for offline backtesting.

    This is used during backtests to:
    1. Avoid live API calls (which cost money and may not have historical data)
    2. Ensure reproducibility (same data = same results)
    3. Enable testing of code changes on historical data

    Usage:
        provider = ReplayProvider(db, logger)
        gex_data = provider.get_gex(symbol, as_of_time)
        flow_data = provider.get_flow(symbol, as_of_time)
        features = provider.get_features(symbol, as_of_time)
    """

    def __init__(self, db: DatabaseDatabase, logger: StructuredLogger):
        self.db = db
        self.logger = logger
        self._cache: Dict[str, Any] = {}

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_row_list(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("rows", "items", "data"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return [item for item in candidate if isinstance(item, dict)]
            return [payload]
        return []

    def get_gex(
        self,
        symbol: str,
        as_of: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Get GEX data from historical snapshot.

        Args:
            symbol: Ticker symbol
            as_of: Point-in-time for data lookup

        Returns:
            List of GEX records (same format as live API)
        """
        cache_key = f"gex:{symbol}:{as_of.isoformat()[:16]}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if isinstance(cached, list):
                return cast(List[Dict[str, Any]], cached)
            return []

        try:
            with self.db.get_session() as session:
                result = (
                    session.query(ExternalSnapshot)
                    .filter(
                        ExternalSnapshot.source == "gex",
                        ExternalSnapshot.symbol == symbol.upper(),
                        ExternalSnapshot.snapshot_time <= as_of,
                    )
                    .order_by(ExternalSnapshot.snapshot_time.desc())
                    .first()
                )

                if result and result.data_json:
                    data = self._to_row_list(result.data_json)
                    self._cache[cache_key] = data
                    return data

                return []
        except Exception as e:
            self.logger.warning("ReplayProvider GEX lookup failed", symbol=symbol, error=str(e))
            return []

    def get_flow(
        self,
        symbol: str,
        as_of: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Get flow data from historical snapshot.

        Args:
            symbol: Ticker symbol
            as_of: Point-in-time for data lookup

        Returns:
            List of flow records (same format as live API)
        """
        cache_key = f"flow:{symbol}:{as_of.isoformat()[:16]}"  # Round to minute
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if isinstance(cached, list):
                return cast(List[Dict[str, Any]], cached)
            return []

        try:
            with self.db.get_session() as session:
                result = (
                    session.query(ExternalSnapshot)
                    .filter(
                        ExternalSnapshot.source == "flow",
                        ExternalSnapshot.symbol == symbol.upper(),
                        ExternalSnapshot.snapshot_time <= as_of,
                    )
                    .order_by(ExternalSnapshot.snapshot_time.desc())
                    .first()
                )

                if result and result.data_json:
                    data = self._to_row_list(result.data_json)
                    self._cache[cache_key] = data
                    return data

                return []
        except Exception as e:
            self.logger.warning("ReplayProvider flow lookup failed", symbol=symbol, error=str(e))
            return []

    def get_features(
        self,
        symbol: str,
        as_of: datetime,
    ) -> Optional[SymbolFeatures]:
        """
        Get pre-computed features from historical snapshot.

        Args:
            symbol: Ticker symbol
            as_of: Point-in-time for data lookup

        Returns:
            SymbolFeatures object or None if not found
        """
        cache_key = f"features:{symbol}:{as_of.isoformat()[:16]}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if isinstance(cached, SymbolFeatures):
                return cached
            return None

        try:
            with self.db.get_session() as session:
                result = (
                    session.query(FeatureSnapshot)
                    .filter(
                        FeatureSnapshot.symbol == symbol.upper(),
                        FeatureSnapshot.as_of_ts <= as_of,
                    )
                    .order_by(FeatureSnapshot.as_of_ts.desc())
                    .first()
                )

                if result:
                    # Reconstruct SymbolFeatures from snapshot blob.
                    features_dict = result.features_json if isinstance(result.features_json, dict) else {}
                    extra_raw = features_dict.get("extra", {})
                    extra = extra_raw if isinstance(extra_raw, dict) else {}

                    features = SymbolFeatures(
                        symbol=result.symbol,
                        price=self._to_float(result.price, self._to_float(features_dict.get("price"), 0.0)),
                        atr_pct=self._to_float(features_dict.get("atr_pct"), 0.0),
                        avg_volume=self._to_float(features_dict.get("avg_volume"), 0.0),
                        intraday_range_pct=self._to_float(features_dict.get("intraday_range_pct"), 0.0),
                        gap_pct=self._to_float(features_dict.get("gap_pct"), 0.0),
                        ema20_slope=self._to_float(features_dict.get("ema20_slope"), 0.0),
                        ema_trend_strength=self._to_float(features_dict.get("ema_trend_strength"), 0.0),
                        distance_from_vwap=self._to_float(features_dict.get("distance_from_vwap"), 0.0),
                        premarket_volume=self._to_float(features_dict.get("premarket_volume"), 0.0),
                        adx=self._to_float(features_dict.get("adx"), 0.0),
                        distance_from_ema20=self._to_float(features_dict.get("distance_from_ema20"), 0.0),
                        prior_day_high=self._to_float(features_dict.get("prior_day_high"), 0.0),
                        prior_day_low=self._to_float(features_dict.get("prior_day_low"), 0.0),
                        bb_upper=self._to_float(features_dict.get("bb_upper"), 0.0),
                        bb_lower=self._to_float(features_dict.get("bb_lower"), 0.0),
                        price_zscore=self._to_float(features_dict.get("price_zscore"), 0.0),
                        flow_zscore=self._to_float(
                            result.flow_zscore, self._to_float(features_dict.get("flow_zscore"), 0.0)
                        ),
                        call_put_ratio=self._to_float(features_dict.get("call_put_ratio"), 0.0),
                        large_sweeps_count=self._to_int(features_dict.get("large_sweeps_count"), 0),
                        aggressive_flow_share=self._to_float(features_dict.get("aggressive_flow_share"), 0.0),
                        last_updated=result.as_of_ts,
                        relative_strength=self._to_float(features_dict.get("relative_strength"), 0.0),
                        flow_bias=self._to_float(features_dict.get("flow_bias"), 0.0),
                        dof_score=self._to_float(features_dict.get("dof_score"), 0.0),
                        atr=self._to_float(result.atr, self._to_float(features_dict.get("atr"), 0.0)),
                        orb_high=self._to_float(features_dict.get("orb_high"), 0.0),
                        orb_low=self._to_float(features_dict.get("orb_low"), 0.0),
                        tfi=self._to_float(result.tfi, self._to_float(features_dict.get("tfi"), 0.0)),
                        net_gex=self._to_float(result.net_gex, self._to_float(features_dict.get("net_gex"), 0.0)),
                        gex_flip_dist=self._to_float(
                            result.gex_flip_dist,
                            self._to_float(features_dict.get("gex_flip_dist"), 0.0),
                        ),
                        frac_diff_close=self._to_float(
                            result.frac_diff,
                            self._to_float(
                                features_dict.get("frac_diff_close", features_dict.get("frac_diff", 0.0)),
                                0.0,
                            ),
                        ),
                        hurst_exponent=self._to_float(
                            result.hurst_exponent,
                            self._to_float(features_dict.get("hurst_exponent"), 0.0),
                        ),
                        ma_dist_50=self._to_float(features_dict.get("ma_dist_50"), 0.0),
                        ma_dist_200=self._to_float(features_dict.get("ma_dist_200"), 0.0),
                        alpha_score=self._to_float(features_dict.get("alpha_score"), 0.0),
                        alpha_rank=self._to_int(features_dict.get("alpha_rank"), 0),
                        ofi=self._to_float(features_dict.get("ofi"), 0.0),
                        high_low_spread=self._to_float(features_dict.get("high_low_spread"), 0.0),
                        extra=cast(Dict[str, Any], extra),
                    )
                    self._cache[cache_key] = features
                    return features

                return None
        except Exception as e:
            self.logger.warning("ReplayProvider features lookup failed", symbol=symbol, error=str(e))
            return None

    def has_data_for_date(self, trade_date: datetime) -> bool:
        """
        Check if we have snapshot data for a given trading day.

        Args:
            trade_date: The trading day to check

        Returns:
            True if any snapshots exist for that date
        """
        try:
            with self.db.get_session() as session:
                from sqlalchemy import func

                # Check for any feature snapshots on that date
                count = (
                    session.query(func.count(FeatureSnapshot.id))
                    .filter(func.date(FeatureSnapshot.as_of_ts) == trade_date.date())
                    .scalar()
                )
                return int(count or 0) > 0
        except Exception as e:
            self.logger.warning("ReplayProvider date check failed", error=str(e))
            return False

    def clear_cache(self) -> None:
        """Clear the in-memory cache (useful between backtest days)."""
        self._cache.clear()
