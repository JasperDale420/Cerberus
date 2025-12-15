from datetime import datetime, timezone
from typing import Dict, List

from src.core.domain import Regime, ScanResult, WatchlistSymbol
from src.core.logger import StructuredLogger
from src.data.pipeline import FeaturePipeline
from src.scanner.profiles import (
    FailedBreakoutProfile,
    FlowMomentumProfile,
    GapProfile,
    IndexMeanReversionProfile,
    ORBScannerProfile,
    ScannerProfile,
    TrendPullbackProfile,
    VWAPReversionProfile,
    VWAPTrendRiderProfile,
)
from src.scanner.universe import UniverseBuilder


class ScanningError(Exception):
    pass


class Scanner:
    """
    Orchestrates the scanning process.
    """

    def __init__(
        self,
        universe_builder: UniverseBuilder,
        feature_pipeline: FeaturePipeline,
        logger: StructuredLogger,
    ):
        self.universe_builder = universe_builder
        self.feature_pipeline = feature_pipeline
        self.logger = logger

        # Initialize profiles
        self.profiles: Dict[str, ScannerProfile] = {
            "vwap_reversion": VWAPReversionProfile(),
            "orb": ORBScannerProfile(),
            "trend_pullback": TrendPullbackProfile(),
            "failed_breakout": FailedBreakoutProfile(),
            "vwap_trend_rider": VWAPTrendRiderProfile(),
            "index_mean_reversion": IndexMeanReversionProfile(),
            "flow_momentum": FlowMomentumProfile(),
            "gap_fill": GapProfile(),
        }

    async def scan(self, regime: Regime = Regime.CHOP) -> ScanResult:
        """
        Runs the scan:
        1. Get Universe
        2. Fetch Features
        3. Apply Profiles
        4. Return Results
        """
        symbols = self.universe_builder.get_universe()
        self.logger.info("Starting scan", universe_size=len(symbols))

        # Fetch features (batch or async loop inside pipeline)
        features_map = await self.feature_pipeline.compute_features(symbols)

        watchlist: List[WatchlistSymbol] = []
        for symbol, features in features_map.items():
            matching_strategies = []
            max_score = 0.0

            for strat_name, profile in self.profiles.items():
                if profile.filter(features):
                    matching_strategies.append(strat_name)
                    # Use the highest score among matching strategies
                    s = profile.score(features, regime)
                    if s > max_score:
                        max_score = s

            if matching_strategies:
                watchlist.append(
                    WatchlistSymbol(
                        symbol=symbol,
                        score=max_score,
                        strategies=matching_strategies,
                        features=features,
                    )
                )

        # Sort watchlist by score descending
        watchlist.sort(key=lambda x: x.score, reverse=True)

        self.logger.info("Scan complete", matches=len(watchlist))

        self.logger.info("Scan complete", matches=len(watchlist))

        return ScanResult(
            generated_at=datetime.now(timezone.utc),  # Metadata timestamp ok here
            regime=regime,
            watchlist=watchlist,
        )
