from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.core.domain import (
    Regime,
    ScanResult,
    StrategyCandidate,
    SymbolFeatures,
    WatchlistSymbol,
)
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
        config: Optional[Dict[str, Any]] = None,
        strategy_profiles: Optional[Dict[str, ScannerProfile]] = None,
    ):
        self.universe_builder = universe_builder
        self.feature_pipeline = feature_pipeline
        self.logger = logger
        self.config = config or {}

        # Initialize profiles (PRD 4.4/4.6: injectable strategy profiles).
        self.profiles: Dict[str, ScannerProfile] = strategy_profiles or {
            "vwap_reversion": VWAPReversionProfile(),
            "orb": ORBScannerProfile(),
            "trend_pullback": TrendPullbackProfile(),
            "failed_breakout": FailedBreakoutProfile(),
            "vwap_trend_rider": VWAPTrendRiderProfile(),
            "index_mean_reversion": IndexMeanReversionProfile(),
            "flow_momentum": FlowMomentumProfile(),
            "gap_fill": GapProfile(),
        }

    async def scan(
        self, regime: Regime = Regime.CHOP, scan_time: Optional[datetime] = None
    ) -> ScanResult:
        """
        Runs the scan:
        1. Get Universe
        2. Fetch Features
        3. Apply Profiles
        4. Return Results
        """
        symbols = self.universe_builder.get_universe()
        universe_size = len(symbols)
        self.logger.info(
            "Starting scan", universe_size=universe_size, regime=regime.value
        )

        # Fetch features (batch or async loop inside pipeline).
        # PRD 4.6 expects `run_scan(regime)` to be callable without extra wiring.
        # Determinism (PRD 11.1): require explicit scan_time or an injected pipeline clock.
        # Optional deterministic fallback: allow config.start_time_utc.
        if scan_time is None:
            clock = getattr(self.feature_pipeline, "clock", None)
            if callable(clock):
                scan_time = clock()
            else:
                start_time = None
                if isinstance(self.config, dict):
                    raw = str(self.config.get("start_time_utc", "") or "").strip()
                    if raw:
                        try:
                            start_time = datetime.fromisoformat(
                                raw.replace("Z", "+00:00")
                            )
                        except Exception:
                            start_time = None
                if start_time is None:
                    self.logger.error(
                        "Scanner.scan requires scan_time (or feature_pipeline.clock) for deterministic behavior",
                        regime=regime.value,
                    )
                    raise ValueError(
                        "Scanner.scan requires scan_time (or feature_pipeline.clock) for deterministic behavior"
                    )
                scan_time = start_time
        if isinstance(scan_time, datetime) and scan_time.tzinfo is None:
            scan_time = scan_time.replace(tzinfo=timezone.utc)
        try:
            features_map = await self.feature_pipeline.compute_features(
                symbols, as_of=scan_time
            )
        except Exception as e:
            self.logger.error(
                "FeaturePipeline.compute_features failed; continuing with empty feature set",
                regime=regime.value,
                universe_size=universe_size,
                error=str(e),
            )
            features_map = {}
        features_returned = len(features_map)

        scanner_cfg = (
            (self.config.get("scanner") or {}) if isinstance(self.config, dict) else {}
        )
        min_price = float(scanner_cfg.get("min_price", 0.0))
        max_price = float(scanner_cfg.get("max_price", float("inf")))
        min_volume = float(scanner_cfg.get("min_volume", 0.0))
        min_atr_pct = float(scanner_cfg.get("min_atr_pct", 0.0))
        max_atr_pct = float(scanner_cfg.get("max_atr_pct", float("inf")))
        top_k_per_strategy = int(scanner_cfg.get("top_k_per_strategy", 10))

        # PRD 4.5: build per-strategy candidate lists then group by symbol.
        candidates_by_strategy: Dict[str, List[StrategyCandidate]] = defaultdict(list)

        baseline_filtered = 0

        def _passes_baseline(f: Any) -> bool:
            try:
                if f.price < min_price or f.price > max_price:
                    return False
                if f.avg_volume < min_volume:
                    return False
                # PRD 4.1/4.3: configurable volatility filter.
                if f.atr_pct < min_atr_pct or f.atr_pct > max_atr_pct:
                    return False
                return True
            except Exception:
                return False

        for symbol, features in features_map.items():
            if not _passes_baseline(features):
                baseline_filtered += 1
                continue
            if not isinstance(features, SymbolFeatures):
                continue
            for strat_name, profile in self.profiles.items():
                if not profile.filter(features):
                    continue
                score = float(profile.score(features, regime))
                candidates_by_strategy[strat_name].append(
                    StrategyCandidate(
                        symbol=symbol,
                        strategy=strat_name,
                        score=score,
                        features=features,
                    )
                )

        # Keep top-K per strategy deterministically.
        pruned: List[StrategyCandidate] = []
        for _strat_name, cands in candidates_by_strategy.items():
            cands_sorted = sorted(cands, key=lambda c: (-c.score, c.symbol))
            pruned.extend(cands_sorted[: max(0, top_k_per_strategy)])

        # Group by symbol.
        by_symbol: Dict[str, Tuple[float, List[str], Any]] = {}
        for c in pruned:
            if c.symbol not in by_symbol:
                by_symbol[c.symbol] = (c.score, [c.strategy], c.features)
            else:
                best, strategies, feats = by_symbol[c.symbol]
                if c.strategy not in strategies:
                    strategies.append(c.strategy)
                by_symbol[c.symbol] = (max(best, c.score), strategies, feats)

        watchlist: List[WatchlistSymbol] = []
        for symbol, (best_score, strategies, feats) in by_symbol.items():
            watchlist.append(
                WatchlistSymbol(
                    symbol=symbol,
                    score=float(best_score),
                    strategies=sorted(strategies),
                    features=feats,
                )
            )

        watchlist.sort(key=lambda w: (-w.score, w.symbol))

        max_watchlist_size = int(
            (self.config.get("scanner") or {}).get("max_watchlist_size", 30)
        )
        # PRD 1.1: Alpaca live data stream practical limit (~30 tickers).
        if max_watchlist_size > 30:
            self.logger.warning(
                "Clamping watchlist size to Alpaca WS limit",
                requested=max_watchlist_size,
                clamped=30,
            )
            max_watchlist_size = 30
        if max_watchlist_size > 0:
            watchlist = watchlist[:max_watchlist_size]

        self.logger.info(
            "Scan complete",
            matches=len(watchlist),
            max_watchlist_size=max_watchlist_size,
            universe_size=universe_size,
            features_returned=features_returned,
            baseline_filtered=baseline_filtered,
            feature_pipeline_metrics=getattr(
                self.feature_pipeline, "last_run_metrics", {}
            ),
        )

        return ScanResult(generated_at=scan_time, regime=regime, watchlist=watchlist)

    def run_scan(
        self, regime: Regime, scan_time: Optional[datetime] = None
    ) -> ScanResult:
        """
        PRD 4.6: `Scanner.run_scan(regime) -> ScanResult`.
        """
        return self.run_scan_sync(regime=regime, scan_time=scan_time)

    def run_scan_symbols(
        self, regime: Regime, scan_time: Optional[datetime] = None
    ) -> List[str]:
        """
        Backwards-compatible helper: returns only the list of symbols.
        """
        result = self.run_scan_sync(regime=regime, scan_time=scan_time)
        return [w.symbol for w in result.watchlist]

    def run_scan_sync(
        self, regime: Regime, scan_time: Optional[datetime] = None
    ) -> ScanResult:
        """
        PRD 4.6-compatible synchronous wrapper.

        Notes:
        - If called from inside an active event loop, use `await run_scan(...)` instead.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.scan(regime=regime, scan_time=scan_time))
        raise RuntimeError(
            "run_scan_sync cannot be called from an active event loop; use await run_scan_async/scan"
        )

    async def run_scan_async(
        self, regime: Regime, scan_time: Optional[datetime] = None
    ) -> ScanResult:
        """
        Async alias over `scan(...)` for callers that prefer PRD naming but run in asyncio.
        """
        return await self.scan(regime=regime, scan_time=scan_time)
