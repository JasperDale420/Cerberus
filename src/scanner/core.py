from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


@dataclass
class _CachedFeature:
    """P5: Cache entry for symbol features with timestamp."""

    features: SymbolFeatures
    cached_at: datetime


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
        from src.scanner.validation import DataValidator

        self.validator = DataValidator(logger)

        # P4: Initialize profiles from config or use defaults
        self.profiles: Dict[str, ScannerProfile] = (
            strategy_profiles
            if strategy_profiles is not None
            else self._build_profiles_from_config()
        )

        # P5: Feature cache with TTL
        self._feature_cache: Dict[str, _CachedFeature] = {}
        scanner_cfg = self.config.get("scanner", {})
        if not isinstance(scanner_cfg, dict):
            scanner_cfg = {}
        try:
            self._cache_ttl_seconds = int(
                scanner_cfg.get("feature_cache_ttl_seconds", 60)
            )
        except (TypeError, ValueError):
            self._cache_ttl_seconds = 60

    def _build_profiles_from_config(self) -> Dict[str, ScannerProfile]:
        """
        P4 fix: Build scanner profiles with configurable thresholds.

        Config format (under scanner.profiles):
            vwap_reversion:
                min_price: 15.0
                min_volume: 100000
                min_sigma: 2.5
            orb:
                min_gap_pct: 0.02
                min_price: 20.0
        """
        profile_cfg = self.config.get("scanner", {}).get("profiles", {})
        if not isinstance(profile_cfg, dict):
            profile_cfg = {}

        def _get(name: str, key: str, default: float) -> float:
            cfg = profile_cfg.get(name, {})
            if not isinstance(cfg, dict):
                return default
            try:
                return float(cfg.get(key, default))
            except (TypeError, ValueError):
                return default

        return {
            "vwap_reversion": VWAPReversionProfile(
                min_price=_get("vwap_reversion", "min_price", 10.0),
                min_volume=_get("vwap_reversion", "min_volume", 0.0),
                min_sigma=_get("vwap_reversion", "min_sigma", 2.0),
            ),
            "orb": ORBScannerProfile(
                min_gap_pct=_get("orb", "min_gap_pct", 0.01),
                min_price=_get("orb", "min_price", 10.0),
            ),
            "trend_pullback": TrendPullbackProfile(
                min_adx=_get("trend_pullback", "min_adx", 25.0),
                max_dist_ema20=_get("trend_pullback", "max_dist_ema20", 0.02),
            ),
            "failed_breakout": FailedBreakoutProfile(
                proximity_pct=_get("failed_breakout", "proximity_pct", 0.02),
            ),
            "vwap_trend_rider": VWAPTrendRiderProfile(
                min_adx=_get("vwap_trend_rider", "min_adx", 20.0),
            ),
            "index_mean_reversion": IndexMeanReversionProfile(
                min_sigma=_get("index_mean_reversion", "min_sigma", 2.0),
            ),
            "flow_momentum": FlowMomentumProfile(
                min_flow_zscore=_get("flow_momentum", "min_flow_zscore", 2.5),
            ),
            "gap_fill": GapProfile(
                min_gap=_get("gap_fill", "min_gap", 0.015),
                max_gap=_get("gap_fill", "max_gap", 0.10),
            ),
        }

    async def scan(
        self, regime: Regime = Regime.CHOP, scan_time: Optional[datetime] = None
    ) -> ScanResult:
        """
        Runs the scan orchestration.
        """
        scan_time = self._resolve_scan_time(scan_time, regime)
        if isinstance(scan_time, datetime) and scan_time.tzinfo is None:
            scan_time = scan_time.replace(tzinfo=timezone.utc)

        symbols = self.universe_builder.get_universe()
        universe_size = len(symbols)
        self.logger.info(
            "Starting scan", universe_size=universe_size, regime=regime.value
        )

        # Stage 1: Fetch Technicals
        features_map = await self._fetch_technicals(
            symbols, scan_time, regime, universe_size
        )
        features_returned = len(features_map)

        # Stage 2: Validate and Filter
        survivors, baseline_filtered = self._apply_data_validation(features_map)

        self.logger.info(
            "Stage 1 technical filter complete",
            total=features_returned,
            passed=len(survivors),
            filtered=baseline_filtered,
        )

        # Stage 3: Fetch Flow
        if survivors:
            await self._fetch_flow_for_survivors(survivors)

        # Stage 4: Score Strategies
        candidates = self._score_strategies(survivors, regime)

        # Stage 5: Build Watchlist
        watchlist = self._build_watchlist(
            candidates, universe_size, features_returned, baseline_filtered
        )

        return ScanResult(generated_at=scan_time, regime=regime, watchlist=watchlist)

    def _resolve_scan_time(
        self, scan_time: Optional[datetime], regime: Regime
    ) -> datetime:
        if scan_time is not None:
            return scan_time

        clock = getattr(self.feature_pipeline, "clock", None)
        if callable(clock):
            return clock()  # type: ignore

        if isinstance(self.config, dict):
            raw = str(self.config.get("start_time_utc", "") or "").strip()
            if raw:
                try:
                    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except Exception:
                    pass

        self.logger.error(
            "Scanner.scan requires scan_time for deterministic behavior",
            regime=regime.value,
        )
        raise ValueError("Scanner.scan requires scan_time")

    async def _fetch_technicals(
        self,
        symbols: List[str],
        scan_time: datetime,
        regime: Regime,
        universe_size: int,
    ) -> Dict[str, SymbolFeatures]:
        """
        P5: Fetch technicals with TTL caching to reduce API calls.
        """
        now = datetime.now(timezone.utc)
        ttl = timedelta(seconds=self._cache_ttl_seconds)

        # Check cache for valid entries
        cached_results: Dict[str, SymbolFeatures] = {}
        symbols_to_fetch: List[str] = []

        for sym in symbols:
            entry = self._feature_cache.get(sym)
            if entry is not None and (now - entry.cached_at) < ttl:
                cached_results[sym] = entry.features
            else:
                symbols_to_fetch.append(sym)

        # Fetch uncached/expired symbols
        if symbols_to_fetch:
            try:
                fresh = await self.feature_pipeline.compute_technicals_only(
                    symbols_to_fetch, as_of=scan_time
                )
                # Update cache
                for sym, features in fresh.items():
                    self._feature_cache[sym] = _CachedFeature(
                        features=features, cached_at=now
                    )
                    cached_results[sym] = features

                if cached_results:
                    self.logger.debug(
                        "Feature cache stats",
                        cache_hits=len(symbols) - len(symbols_to_fetch),
                        cache_misses=len(symbols_to_fetch),
                        fetched=len(fresh),
                    )
            except Exception as e:
                self.logger.error(
                    "FeaturePipeline failed",
                    regime=regime.value,
                    universe_size=universe_size,
                    error=str(e),
                )
                # Return whatever we have cached
                return cached_results

        return cached_results

    def _apply_data_validation(
        self, features_map: Dict[str, SymbolFeatures]
    ) -> Tuple[Dict[str, SymbolFeatures], int]:
        survivors = {}
        baseline_filtered = 0

        scanner_cfg = (
            (self.config.get("scanner") or {}) if isinstance(self.config, dict) else {}
        )

        # Extract filter params
        params = {
            "min_price": float(scanner_cfg.get("min_price", 0.0)),
            "max_price": float(scanner_cfg.get("max_price", float("inf"))),
            "min_volume": float(scanner_cfg.get("min_volume", 0.0)),
            "min_atr_pct": float(scanner_cfg.get("min_atr_pct", 0.0)),
            "max_atr_pct": float(scanner_cfg.get("max_atr_pct", float("inf"))),
        }

        for symbol, features in features_map.items():
            if self.validator.validate_technicals(features, **params):
                survivors[symbol] = features
            else:
                baseline_filtered += 1

        return survivors, baseline_filtered

    async def _fetch_flow_for_survivors(
        self, survivors: Dict[str, SymbolFeatures]
    ) -> None:
        try:
            await self.feature_pipeline.append_flow_features(survivors)
        except Exception as e:
            self.logger.error(
                "FeaturePipeline.append_flow_features failed",
                error=str(e),
            )

    def _score_strategies(
        self, survivors: Dict[str, SymbolFeatures], regime: Regime
    ) -> List[StrategyCandidate]:
        candidates = []
        for symbol, features in survivors.items():
            if not isinstance(features, SymbolFeatures):
                continue
            for strat_name, profile in self.profiles.items():
                if profile.filter(features):
                    score = float(profile.score(features, regime))
                    candidates.append(
                        StrategyCandidate(
                            symbol=symbol,
                            strategy=strat_name,
                            score=score,
                            features=features,
                        )
                    )
        return candidates

    def _build_watchlist(
        self,
        candidates: List[StrategyCandidate],
        universe_size: int,
        features_returned: int,
        baseline_filtered: int,
    ) -> List[WatchlistSymbol]:
        scanner_cfg = (
            (self.config.get("scanner") or {}) if isinstance(self.config, dict) else {}
        )
        top_k = int(scanner_cfg.get("top_k_per_strategy", 10))

        # Group by strategy and prune
        by_viz: Dict[str, List[StrategyCandidate]] = defaultdict(list)
        for c in candidates:
            by_viz[c.strategy].append(c)

        pruned = []
        for cands in by_viz.values():
            cands_sorted = sorted(cands, key=lambda c: (-c.score, c.symbol))
            pruned.extend(cands_sorted[: max(0, top_k)])

        # Group by symbol
        by_symbol: Dict[str, Tuple[float, List[str], Any]] = {}
        for c in pruned:
            if c.symbol not in by_symbol:
                by_symbol[c.symbol] = (c.score, [c.strategy], c.features)
            else:
                best, strategies, feats = by_symbol[c.symbol]
                if c.strategy not in strategies:
                    strategies.append(c.strategy)
                by_symbol[c.symbol] = (max(best, c.score), strategies, feats)

        watchlist = [
            WatchlistSymbol(
                symbol=sym,
                score=float(score),
                strategies=sorted(strats),
                features=feats,
            )
            for sym, (score, strats, feats) in by_symbol.items()
        ]
        watchlist.sort(key=lambda w: (-w.score, w.symbol))

        # M3 fix: Configurable watchlist size with documented default
        # PRD recommends max 30 symbols for manageable tracking; larger values may impact performance
        max_size = int(scanner_cfg.get("max_watchlist_size", 30))
        default_cap = 50  # Hard cap for safety
        if max_size > default_cap:
            self.logger.warning(
                "Clamping watchlist size to safety cap",
                requested=max_size,
                clamped=default_cap,
                hint="PRD recommends 30 or fewer symbols for manageable tracking",
            )
            max_size = default_cap

        if max_size > 0:
            watchlist = watchlist[:max_size]

        self.logger.info(
            "Scan complete",
            matches=len(watchlist),
            max_watchlist_size=max_size,
            universe_size=universe_size,
            features_returned=features_returned,
            baseline_filtered=baseline_filtered,
            feature_pipeline_metrics=getattr(
                self.feature_pipeline, "last_run_metrics", {}
            ),
        )
        return watchlist

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
