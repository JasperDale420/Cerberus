import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import pytz  # type: ignore

from src.core.domain import SymbolFeatures
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.data.api_client import CentralApiClient
from src.data.calculator import FeatureCalculator
from src.data.fetcher import DataFetcher
from src.data.unusual_whales import UnusualWhalesClient

if TYPE_CHECKING:
    from src.data.snapshot_manager import SnapshotManager

US_EASTERN = pytz.timezone("US/Eastern")
UTC_ZERO_STR = "+00:00"


class FeaturePipeline:
    """
    Fetches data and computes features for symbols.
    """

    def __init__(
        self,
        alpaca_client: Optional[AlpacaClient],
        unusual_whales_client: UnusualWhalesClient,
        logger: StructuredLogger,
        config: Optional[Dict[str, Any]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        snapshot_manager: Optional["SnapshotManager"] = None,
        central_api_client: Optional[CentralApiClient] = None,
    ):
        self.alpaca_client = alpaca_client
        self.unusual_whales_client = unusual_whales_client
        self.logger = logger
        self.config = config or {}
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.snapshot_manager = snapshot_manager

        fp_cfg = (self.config.get("feature_pipeline") or {}) if isinstance(self.config, dict) else {}
        self.daily_volume_lookback_days = int(fp_cfg.get("daily_volume_lookback_days", 20))
        self.max_concurrency = int(fp_cfg.get("max_concurrency", 6))

        # Snapshot capture settings
        snap_cfg = self.config.get("snapshots", {}) if isinstance(self.config, dict) else {}
        self.snapshots_enabled = bool(snap_cfg.get("enabled", False))

        # New Collaborators
        self.fetcher = DataFetcher(
            alpaca_client,
            unusual_whales_client,
            logger,
            central_api_client=central_api_client,
            config=config,
            clock=self.clock,
        )
        self.calculator = FeatureCalculator()

    def _calculate_fetch_window(self, as_of: datetime) -> tuple[datetime, datetime]:
        end = as_of
        if isinstance(end, datetime) and end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        # Use 24h lookback to ensure prior day coverage for indicators
        start = end - timedelta(hours=24)
        return start, end

    def _extract_closes(self, bars_data: List[Any]) -> List[float]:
        """Extract close prices from bar objects/dicts (fast, allocation-light)."""
        closes: List[float] = []
        for bar in bars_data:
            if hasattr(bar, "c"):
                value = getattr(bar, "c", 0)
            elif isinstance(bar, dict):
                value = bar.get("c", 0)
            else:
                value = 0
            closes.append(float(value or 0))
        return closes

    async def _fetch_bars_wrapper(self, sym: str, as_of: datetime) -> tuple[List[Any], Dict[str, int]]:
        start, end = self._calculate_fetch_window(as_of)
        bars_data, fetch_metrics = await self.fetcher.fetch_bars(sym, start, end, "1Min")
        return bars_data, fetch_metrics

    async def _fetch_supplementary_data(
        self, sym: str, end: datetime
    ) -> tuple[Optional[float], tuple[float, float, float]]:
        avg_daily_volume = await asyncio.to_thread(
            self.fetcher.fetch_avg_daily_volume,
            sym,
            end,
            self.daily_volume_lookback_days,
        )

        prior_stats = await asyncio.to_thread(
            self.fetcher.fetch_prior_day_stats,
            sym,
            end,
        )
        return avg_daily_volume, prior_stats

    def _build_symbol_features(
        self,
        sym: str,
        tech_result: Any,
        avg_daily_volume: Optional[float],
        prior_stats: tuple[float, float, float],
        bars_data: List[Any],
        end: datetime,
    ) -> SymbolFeatures:
        p_high, p_low, p_close = prior_stats

        # Fix floating point equality check logic
        if abs(tech_result.prior_day_high) < 1e-9 or abs(tech_result.prior_day_low) < 1e-9:
            if p_high > 0:
                prior_day_high = p_high
                prior_day_low = p_low
            else:
                prior_day_high = tech_result.prior_day_high
                prior_day_low = tech_result.prior_day_low
        else:
            prior_day_high = tech_result.prior_day_high
            prior_day_low = tech_result.prior_day_low

        session_open = self.calculator.calculate_session_open_price(
            bars_data if isinstance(bars_data, list) else [], end
        )

        gap_pct = tech_result.gap_pct
        if p_close > 0 and session_open > 0:
            gap_pct = (session_open - p_close) / p_close

        # Create features with NEUTRAL flow data
        return SymbolFeatures(
            symbol=sym,
            price=tech_result.price,
            atr_pct=tech_result.atr_pct,
            avg_volume=(float(avg_daily_volume) if avg_daily_volume is not None else float(tech_result.volume)),
            intraday_range_pct=tech_result.intraday_range_pct,
            gap_pct=gap_pct,
            ema20_slope=tech_result.ema20_slope,
            ema_trend_strength=abs(tech_result.ema20_slope),
            distance_from_vwap=tech_result.distance_from_vwap,
            premarket_volume=tech_result.premarket_volume,
            adx=tech_result.adx,
            distance_from_ema20=tech_result.distance_from_ema20,
            prior_day_high=prior_day_high,
            prior_day_low=prior_day_low,
            atr=tech_result.atr,
            bb_upper=tech_result.bb_upper,
            bb_lower=tech_result.bb_lower,
            price_zscore=tech_result.price_zscore,
            # Neutral flow details
            flow_zscore=0.0,
            call_put_ratio=0.0,
            large_sweeps_count=0,
            aggressive_flow_share=0.0,
            tfi=tech_result.tfi,
            last_updated=(
                tech_result.timestamp
                if isinstance(tech_result.timestamp, datetime)
                else datetime.fromisoformat(str(tech_result.timestamp).replace("Z", UTC_ZERO_STR))
            ),
            relative_strength=getattr(tech_result, "relative_strength", 0.0),
            flow_bias=0.0,
            dof_score=0.0,
            orb_high=tech_result.orb_high,
            orb_low=tech_result.orb_low,
            net_gex=0.0,
            gex_flip_dist=0.0,
            frac_diff_close=tech_result.frac_diff_close,
            hurst_exponent=tech_result.hurst_exponent,
            extra={
                "flow_raw_count": 0,
                "flow_bias": 0.0,
                "dof_score": 0.0,
                "volatility": tech_result.atr_pct,
                "last_bar_volume": float(tech_result.volume),
                "avg_daily_volume_days": int(self.daily_volume_lookback_days),
            },
        )

    async def _process_single_symbol(
        self,
        symbol: str,
        as_of: datetime,
        sem: asyncio.Semaphore,
        benchmark_closes: Optional[List[float]] = None,
    ) -> tuple[str, Optional[SymbolFeatures], Dict[str, int]]:
        local: Dict[str, int] = {
            "features_ok": 0,
            "alpaca_fetch_fail": 0,
            "alpaca_no_bars": 0,
            "technicals_fail": 0,
        }
        sym = str(symbol).strip().upper()

        async with sem:
            try:
                # 1. Fetch Bars
                bars_data, fetch_metrics = await self._fetch_bars_wrapper(sym, as_of)
                for k, v in fetch_metrics.items():
                    local[k] = int(local.get(k, 0)) + int(v)

                if not bars_data:
                    self.logger.warning("No bars found for technicals", symbol=sym)
                    local["alpaca_no_bars"] += 1
                    return sym, None, local

                # 2. Compute Technicals
                tech_result = self.calculator.compute_technicals(bars_data)
                if not tech_result:
                    local["technicals_fail"] += 1
                    return sym, None, local

                # 2b. Compute Relative Strength if benchmark available
                closes: Optional[List[float]] = None
                if benchmark_closes and bars_data:
                    # Performance: extract closes once to avoid a second full pass later.
                    closes = self._extract_closes(bars_data)
                    tech_result.relative_strength = self.calculator.calculate_relative_strength(
                        closes, benchmark_closes
                    )

                # 3. Fetch Supplementary Data
                _, end = self._calculate_fetch_window(as_of)
                avg_daily_volume, prior_stats = await self._fetch_supplementary_data(sym, end)

                # 4. Fetch Trades for TFI (last 5 minutes for high-res microstructure)
                trades_start = as_of - timedelta(minutes=5)
                trades_data, _ = await self.fetcher.fetch_trades(sym, trades_start, as_of)
                tech_result.tfi = self.calculator.calculate_tfi(trades_data)

                # 4b. Compute Statistical Alpha (FracDiff & Hurst)
                if bars_data:
                    if closes is None:
                        closes = self._extract_closes(bars_data)
                    tech_result.frac_diff_close = self.calculator.apply_frac_diff(closes, d=0.4)
                    tech_result.hurst_exponent = self.calculator.calculate_hurst_exponent(closes)

                # 5. Build Features
                feat = self._build_symbol_features(sym, tech_result, avg_daily_volume, prior_stats, bars_data, end)

                local["features_ok"] += 1
                return sym, feat, local
            except Exception as e:
                local["technicals_fail"] += 1
                self.logger.error(
                    "Feature pipeline symbol processing failed",
                    symbol=sym,
                    error=str(e),
                )
                return sym, None, local

    async def compute_technicals_only(
        self, symbols: List[str], as_of: Optional[datetime] = None
    ) -> Dict[str, SymbolFeatures]:
        """
        Stage 1: Compute technical features only (Alpaca data).
        Returns features with neutral/empty flow data.
        """
        features: Dict[str, SymbolFeatures] = {}
        if as_of is None:
            raise ValueError("FeaturePipeline.compute_technicals_only requires as_of for deterministic behavior")

        # Initialize metrics for this run
        metrics: Dict[str, int] = {
            "symbols_total": int(len(symbols)),
            "features_ok": 0,
            "alpaca_fetch_fail": 0,
            "alpaca_no_bars": 0,
            "technicals_fail": 0,
            "uw_fetch_fail": 0,  # explicit 0 here
            "cache_hits": 0,
            "incremental_fetches": 0,
        }

        import asyncio

        sem = asyncio.Semaphore(max(1, int(self.max_concurrency)))

        # Stage 1a: Fetch Benchmark (SPY) for Relative Strength
        benchmark_closes: List[float] = []
        try:
            start, end = self._calculate_fetch_window(as_of)
            spy_bars, _ = await self.fetcher.fetch_bars("SPY", start, end, "1Min")
            benchmark_closes = [float(b.c if hasattr(b, "c") else b.get("c", 0)) for b in spy_bars]
        except Exception as e:
            self.logger.warning("Failed to fetch benchmark SPY bars", error=str(e))

        # Use symbols list directly
        results = await asyncio.gather(*[self._process_single_symbol(s, as_of, sem, benchmark_closes) for s in symbols])
        for sym, feat, local in results:
            for k, v in local.items():
                metrics[k] = int(metrics.get(k, 0)) + int(v)
            if feat is not None:
                features[sym] = feat

        # Store intermediate metrics? Or merge later.
        # We'll merge in the wrapper.
        self.last_run_metrics = dict(metrics)  # partial update
        return features

    async def append_flow_features(self, features_map: Dict[str, SymbolFeatures]) -> Dict[str, SymbolFeatures]:
        """
        Stage 2: Fetch and append UW options flow data for existing features.
        """
        if not features_map:
            return {}

        import asyncio

        # UW API is strict (429s observed). Enforce serial fetching + delay.
        sem_flow = asyncio.Semaphore(1)

        metrics = dict(getattr(self, "last_run_metrics", {}))
        metrics["uw_fetch_fail"] = 0  # reset for this stage increment

        # Check explicit enabled flag (default to True/enabled if missing to preserve legacy behavior,
        # unless we want to default to False. The plan said check config.)
        # Config structure: self.config["unusual_whales"]["enabled"]
        uw_cfg = self.config.get("unusual_whales") or {}
        uw_enabled = bool(uw_cfg.get("enabled", True))

        if not uw_enabled:
            # PRD 10.6: External flow integration disabled.
            # We skip fetching but must ensure fields are zeroed (already done in Stage 1 init).
            self.logger.info("Unusual Whales flow integration disabled; skipping fetch.")
            return features_map

        async def _enrich_one(feat: SymbolFeatures) -> SymbolFeatures:
            sym = feat.symbol
            local_fail = 0

            async with sem_flow:
                # Add delay to respect rate limits (approx 2 req/sec safe limit?)
                await asyncio.sleep(0.5)
                try:
                    # we need a date for the flow fetch. feature.last_updated should be consistent with as_of.
                    date_str = feat.last_updated.strftime("%Y-%m-%d")
                    flow_data = await self.fetcher.fetch_flow(sym, date_str)

                    # Also fetch Greek Exposure for GEX
                    gex_data = await self.fetcher.fetch_gex(sym)
                except Exception:
                    local_fail = 1
                    # Log already handled in fetcher but we can log context here too if needed
                    flow_data = []

            (c_p_ratio, f_zscore, sw_count, agg_share, f_bias) = self.calculator.compute_flow_metrics(flow_data)
            d_score = abs(float(f_bias)) * float(agg_share)
            if int(sw_count) > 5:
                d_score *= 1.2
            d_score = max(0.0, min(1.0, d_score))

            # Compute GEX metrics
            net_gex, gex_flip_dist = self.calculator.calculate_gex_metrics(gex_data, feat.price)

            # Update feature object (dataclass is mutable-ish or we allow direct field update)
            # Python dataclasses are mutable by default.
            feat.flow_zscore = f_zscore
            feat.call_put_ratio = c_p_ratio
            feat.large_sweeps_count = sw_count
            feat.aggressive_flow_share = agg_share
            feat.flow_bias = f_bias
            feat.dof_score = d_score
            feat.net_gex = net_gex
            feat.gex_flip_dist = gex_flip_dist

            if feat.extra is None:
                feat.extra = {}
            feat.extra["flow_raw_count"] = len(flow_data) if flow_data else 0
            feat.extra["flow_bias"] = f_bias
            feat.extra["dof_score"] = d_score

            if local_fail:
                metrics["uw_fetch_fail"] = int(metrics.get("uw_fetch_fail", 0)) + 1

            # Persist external snapshots if enabled
            if self.snapshots_enabled and self.snapshot_manager:
                now = self.clock()
                if flow_data:
                    self.snapshot_manager.persist_external_snapshot(
                        source="flow",
                        symbol=sym,
                        snapshot_time=now,
                        data=flow_data,
                    )
                if gex_data:
                    self.snapshot_manager.persist_external_snapshot(
                        source="gex",
                        symbol=sym,
                        snapshot_time=now,
                        data=gex_data,
                    )

            return feat

        # Only fetch for symbols present in map
        tasks = [_enrich_one(f) for f in features_map.values()]
        await asyncio.gather(*tasks)

        # Update metrics
        try:
            # Add stage 2 duration to whatever Stage 1 was? Or just track total in wrapper.
            # We'll update global metrics mainly related to fetch failures.
            self.last_run_metrics = dict(metrics)
        except Exception:
            pass

        return features_map

    async def compute_features(self, symbols: List[str], as_of: Optional[datetime] = None) -> Dict[str, SymbolFeatures]:
        """
        Computes features for a list of symbols (Full Pipeline).
        """
        # Run Stage 1
        features = await self.compute_technicals_only(symbols, as_of=as_of)

        # Run Stage 2 (on ALL symbols, preserving legacy behavior)
        # Using a copy of keys to pass list is implied by append_flow_features taking the dict
        await self.append_flow_features(features)

        # Persist feature snapshots if enabled
        if self.snapshots_enabled and self.snapshot_manager:
            now = as_of or self.clock()
            for feat in features.values():
                self.snapshot_manager.persist_feature_snapshot(features=feat, as_of_ts=now)

        # Finalize metrics
        if hasattr(self, "last_run_metrics"):
            self.logger.info("FeaturePipeline summary", **self.last_run_metrics)

        return features
