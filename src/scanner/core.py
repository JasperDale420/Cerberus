from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.config.models import PairTradingConfig
from src.core.domain import (
    Bar,
    Regime,
    ScanResult,
    StrategyCandidate,
    SymbolFeatures,
    WatchlistSymbol,
)
from src.core.logger import StructuredLogger
from src.data.pipeline import FeaturePipeline
from src.scanner.pair_scanner import PairScanner
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
from src.scanner.ranking import RankingEngine
from src.scanner.universe import UniverseBuilder


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
            strategy_profiles if strategy_profiles is not None else self._build_profiles_from_config()
        )
        self.ranking_engine = RankingEngine(logger)

        # Initialize Pair trading config and engine
        risk_cfg_dict = self.config.get("risk", {})
        pair_cfg_dict = risk_cfg_dict.get("pair_trading", {})
        self.pair_config = PairTradingConfig(**pair_cfg_dict)
        self.pair_scanner = PairScanner(self.pair_config, logger)

        # Phase 3 Optimization: Pair discovery cache
        self._discovered_pairs: List[Dict[str, Any]] = []
        self._last_pair_discovery_time: Optional[datetime] = None
        self._discovery_interval = timedelta(
            minutes=int(self.config.get("scanner", {}).get("pair_discovery_interval_minutes", 60))
        )

        # P5: Feature cache with TTL + LRU eviction (H1 memory audit fix)
        # Uses OrderedDict for LRU ordering; evicts oldest when maxsize exceeded
        from collections import OrderedDict

        self._feature_cache: OrderedDict[str, _CachedFeature] = OrderedDict()
        scanner_cfg = self.config.get("scanner", {})
        if not isinstance(scanner_cfg, dict):
            scanner_cfg = {}
        try:
            self._cache_ttl_seconds = int(scanner_cfg.get("feature_cache_ttl_seconds", 60))
        except (TypeError, ValueError):
            self._cache_ttl_seconds = 60
        try:
            self._cache_maxsize = int(scanner_cfg.get("feature_cache_maxsize", 1000))
        except (TypeError, ValueError):
            self._cache_maxsize = 1000

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

    async def scan(self, regime: Regime = Regime.CHOP, scan_time: Optional[datetime] = None) -> ScanResult:
        """
        Runs the scan orchestration.
        """
        scan_time = self._resolve_scan_time(scan_time, regime)
        if isinstance(scan_time, datetime) and scan_time.tzinfo is None:
            scan_time = scan_time.replace(tzinfo=timezone.utc)

        symbols = self.universe_builder.get_universe()
        universe_size = len(symbols)
        self.logger.info("Starting scan", universe_size=universe_size, regime=regime.value)

        # Stage 1: Fetch Technicals
        features_map = await self._fetch_technicals(symbols, scan_time, regime, universe_size)
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

        # Stage 4: Ranking Engine (Cross-Sectional Alpha)
        if survivors:
            self.ranking_engine.rank_symbols(list(survivors.values()))

        # Stage 5: Score Strategies
        candidates = self._score_strategies(survivors, regime)

        # Stage 6: Build Watchlist
        watchlist = self._build_watchlist(
            candidates,
            universe_size,
            features_returned,
            baseline_filtered,
            survivors,
            regime,
        )

        # Stage 7: Pair Trading (Dynamic Pair Selection)
        if self.pair_config.enabled and survivors:
            await self._scan_pairs(survivors, scan_time, watchlist)

        return ScanResult(generated_at=scan_time, regime=regime, watchlist=watchlist)

    def _resolve_scan_time(self, scan_time: Optional[datetime], regime: Regime) -> datetime:
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

    def _calculate_fetch_window(self, as_of: datetime) -> Tuple[datetime, datetime]:
        end = as_of
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        # Expand lookback to 24 hours to ensure we have prior day data for indicators
        # like prior_day_high, EMA20, and ATR.
        start = end - timedelta(hours=24)
        return start, end

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
        # Backtest Fix: use scan_time for TTL check, not real-time now()
        now = scan_time if scan_time.tzinfo else scan_time.replace(tzinfo=timezone.utc)
        ttl = timedelta(seconds=self._cache_ttl_seconds)

        # Check cache for valid entries
        cached_results: Dict[str, SymbolFeatures] = {}
        symbols_to_fetch: List[str] = []

        for sym in symbols:
            entry = self._feature_cache.get(sym)
            if entry is not None and (now - entry.cached_at) < ttl:
                cached_results[sym] = entry.features
                # LRU: move accessed item to end
                self._feature_cache.move_to_end(sym)
            else:
                symbols_to_fetch.append(sym)

        # Fetch uncached/expired symbols
        if symbols_to_fetch:
            try:
                fresh = await self.feature_pipeline.compute_technicals_only(symbols_to_fetch, as_of=scan_time)
                # Update cache with LRU eviction
                for sym, features in fresh.items():
                    self._feature_cache[sym] = _CachedFeature(features=features, cached_at=now)
                    self._feature_cache.move_to_end(sym)
                    cached_results[sym] = features

                # Evict oldest entries if over maxsize
                while len(self._feature_cache) > self._cache_maxsize:
                    evicted_sym, _ = self._feature_cache.popitem(last=False)
                    self.logger.debug("Evicted stale cache entry", symbol=evicted_sym)

                if cached_results:
                    self.logger.debug(
                        "Feature cache stats",
                        cache_hits=len(symbols) - len(symbols_to_fetch),
                        cache_misses=len(symbols_to_fetch),
                        fetched=len(fresh),
                        cache_size=len(self._feature_cache),
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

    def _apply_data_validation(self, features_map: Dict[str, SymbolFeatures]) -> Tuple[Dict[str, SymbolFeatures], int]:
        survivors = {}
        baseline_filtered = 0

        scanner_cfg = (self.config.get("scanner") or {}) if isinstance(self.config, dict) else {}

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

    async def _fetch_flow_for_survivors(self, survivors: Dict[str, SymbolFeatures]) -> None:
        try:
            await self.feature_pipeline.append_flow_features(survivors)
        except Exception as e:
            self.logger.error(
                "FeaturePipeline.append_flow_features failed",
                error=str(e),
            )

    def _score_strategies(self, survivors: Dict[str, SymbolFeatures], regime: Regime) -> List[StrategyCandidate]:
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
        survivors: Dict[str, SymbolFeatures],
        regime: Regime,
    ) -> List[WatchlistSymbol]:
        scanner_cfg = (self.config.get("scanner") or {}) if isinstance(self.config, dict) else {}
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

        # Add global strategies from routing for the current regime
        routing_cfg = self.config.get("strategy_routing", {})
        regime_key = regime.value if hasattr(regime, "value") else str(regime)
        global_strategies = list(routing_cfg.get(regime_key, []))

        for c in pruned:
            if c.symbol not in by_symbol:
                # Merge profile-selected strategies with global strategies
                strats = list(set(global_strategies + [c.strategy]))
                by_symbol[c.symbol] = (c.score, strats, c.features)
            else:
                best, strategies, feats = by_symbol[c.symbol]
                if c.strategy not in strategies:
                    strategies.append(c.strategy)
                by_symbol[c.symbol] = (max(best, c.score), strategies, feats)

        # Handle case where some symbols passed baseline but had no profile matches
        # We still want to add them if global strategies (like fusion_v1) are enabled
        potential_symbols = set()
        for c in candidates:
            potential_symbols.add(c.symbol)

        # If we have global strategies, ensure baseline-passing symbols are considered
        # even if they didn't match a specific 'profile' (Stage 4)
        if global_strategies:
            for sym, feats in survivors.items():
                if sym not in by_symbol:
                    # Give them a baseline score (relative_strength or 0)
                    score = float(getattr(feats, "relative_strength", 0.0))
                    by_symbol[sym] = (score, list(global_strategies), feats)

        watchlist_all = [
            WatchlistSymbol(
                symbol=sym,
                score=float(score),
                strategies=sorted(strats),
                features=feats,
            )
            for sym, (score, strats, feats) in by_symbol.items()
        ]

        # Apply Top-N Alpha Rank gating if enabled
        alpha_rank_limit = int(scanner_cfg.get("alpha_rank_limit", 0))
        if alpha_rank_limit > 0:
            watchlist_all = [w for w in watchlist_all if getattr(w.features, "alpha_rank", 999) <= alpha_rank_limit]
            self.logger.info(
                "Applied Alpha Rank gating",
                limit=alpha_rank_limit,
                remaining=len(watchlist_all),
            )

        # Sort by AlphaScore descending (falling back to strategy score)
        watchlist = sorted(
            watchlist_all,
            key=lambda w: (
                -getattr(w.features, "alpha_score", 0.0),
                -w.score,
                w.symbol,
            ),
        )

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
            feature_pipeline_metrics=getattr(self.feature_pipeline, "last_run_metrics", {}),
        )
        return watchlist

    def run_scan(self, regime: Regime, scan_time: Optional[datetime] = None) -> ScanResult:
        """
        PRD 4.6: `Scanner.run_scan(regime) -> ScanResult`.
        """
        return self.run_scan_sync(regime=regime, scan_time=scan_time)

    def run_scan_sync(self, regime: Regime, scan_time: Optional[datetime] = None) -> ScanResult:
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
        raise RuntimeError("run_scan_sync cannot be called from an active event loop; use await scan()")

    async def _scan_pairs(
        self,
        survivors: Dict[str, SymbolFeatures],
        scan_time: datetime,
        watchlist: List[WatchlistSymbol],
    ) -> None:
        """
        Dynamic Pair Selection and Spread Signal Generation.
        Optimization: Finding pairs (cointegration) is slow, so we cache them
        and only re-discover periodically. Z-scores are updated every scan.
        """
        import asyncio

        import pandas as pd

        from src.core.domain import OrderSide

        # 1. Decide if we need a new discovery run
        should_discover = (
            self._last_pair_discovery_time is None
            or (scan_time - self._last_pair_discovery_time) >= self._discovery_interval
        )

        if should_discover:
            # 1a. Select top survivors to avoid O(N^2)
            all_survivors = list(survivors.values())
            # Sort by volume to ensure we pick liquid pairs
            top_n_feat = sorted(all_survivors, key=lambda x: x.avg_volume, reverse=True)[:50]
            symbols = [s.symbol for s in top_n_feat]

            if len(symbols) >= 2:
                self.logger.info("Discovering cointegrated pairs", candidate_count=len(symbols))
                lookback = self.pair_config.lookback_days
                window_start = scan_time - timedelta(days=int(lookback * 1.5) + 10)

                tasks = [
                    self.feature_pipeline.fetcher.fetch_bars(sym, window_start, scan_time, timeframe="1Day")
                    for sym in symbols
                ]
                results: List[Any] = await asyncio.gather(*tasks, return_exceptions=True)

                price_data: Dict[str, pd.Series] = {}
                for sym, res in zip(symbols, results, strict=False):
                    if isinstance(res, BaseException) or not res:
                        continue
                    if not isinstance(res, tuple) or len(res) != 2:
                        continue
                    bars_raw, _ = res
                    if not isinstance(bars_raw, list):
                        continue
                    bars = [b for b in bars_raw if isinstance(b, Bar)]
                    if not bars:
                        continue

                    s = pd.Series(
                        data=[float(b.close) for b in bars],
                        index=[b.time for b in bars],
                        dtype=float,
                    )
                    if len(s) >= lookback:
                        price_data[sym] = s

                if len(price_data) >= 2:
                    matrix = pd.DataFrame(price_data).ffill().dropna()
                    if not matrix.empty and len(matrix.columns) >= 2:
                        self._discovered_pairs = self.pair_scanner.find_pairs(matrix)
                        self._last_pair_discovery_time = scan_time
                        self.logger.info("Pair discovery complete", found=len(self._discovered_pairs))

        # 2. Update Z-scores for all discovered pairs using current survivor prices
        for pair in self._discovered_pairs:
            s1, s2 = pair["symbol_a"], pair["symbol_b"]

            # Both symbols must be in current survivors to generate a signal
            if s1 not in survivors or s2 not in survivors:
                continue

            p1 = survivors[s1].price
            p2 = survivors[s2].price

            # spread = y - hedge * x - alpha
            # pair_scanner ensures y is s1, x is s2
            spread = p1 - (pair["hedge_ratio"] * p2) - pair.get("alpha", 0.0)
            z_score = (spread - pair["spread_mean"]) / pair["spread_std"]

            if abs(z_score) >= self.pair_config.entry_zscore:
                side_1 = OrderSide.SELL if z_score > 0 else OrderSide.BUY
                side_2 = OrderSide.BUY if z_score > 0 else OrderSide.SELL

                pair_id = f"pair_{s1}_{s2}_{scan_time.strftime('%H%M%S')}"

                metadata = [
                    (s1, side_1, s2, pair["hedge_ratio"], z_score, p2),
                    (s2, side_2, s1, 1.0, -z_score, p1),
                ]

                for sym, side, other, h_ratio, zs, other_price in metadata:
                    feat = survivors[sym]
                    feat.extra["pair_id"] = pair_id
                    feat.extra["pair_side"] = side.value
                    feat.extra["pair_partner"] = other
                    feat.extra["pair_partner_price"] = other_price
                    feat.extra["hedge_ratio"] = h_ratio
                    feat.extra["spread_zscore"] = zs

                    # Ensure in watchlist
                    existing = next((w for w in watchlist if w.symbol == sym), None)
                    if existing:
                        if "pair_trading" not in existing.strategies:
                            existing.strategies.append("pair_trading")
                    else:
                        watchlist.append(
                            WatchlistSymbol(
                                symbol=sym,
                                score=abs(zs),
                                strategies=["pair_trading"],
                                features=feat,
                            )
                        )

                self.logger.info(
                    "Pair signal generated",
                    pair_id=pair_id,
                    z_score=z_score,
                    sym_a=s1,
                    sym_b=s2,
                )
