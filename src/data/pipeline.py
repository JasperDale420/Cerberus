import asyncio
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pytz  # type: ignore

from src.core.domain import SymbolFeatures
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.data.calculator import FeatureCalculator
from src.data.fetcher import DataFetcher
from src.data.unusual_whales import UnusualWhalesClient

US_EASTERN = pytz.timezone("US/Eastern")
UTC_ZERO_STR = "+00:00"


class FeaturePipeline:
    """
    Fetches data and computes features for symbols.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        unusual_whales_client: UnusualWhalesClient,
        logger: StructuredLogger,
        config: Optional[Dict[str, Any]] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.alpaca_client = alpaca_client
        self.unusual_whales_client = unusual_whales_client
        self.logger = logger
        self.config = config or {}
        self.clock = clock or (lambda: datetime.now(timezone.utc))

        fp_cfg = (
            (self.config.get("feature_pipeline") or {})
            if isinstance(self.config, dict)
            else {}
        )
        self.daily_volume_lookback_days = int(
            fp_cfg.get("daily_volume_lookback_days", 20)
        )
        self.max_concurrency = int(fp_cfg.get("max_concurrency", 6))

        # New Collaborators
        self.fetcher = DataFetcher(
            alpaca_client, unusual_whales_client, logger, config, self.clock
        )
        self.calculator = FeatureCalculator()

    def _calculate_fetch_window(self, as_of: datetime) -> tuple[datetime, datetime]:
        end = as_of
        if isinstance(end, datetime) and end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        end_et = end.astimezone(US_EASTERN)
        start_day = end_et.date()
        if end_et.time() < time(4, 0):
            start_day = start_day - timedelta(days=1)
        start_et = US_EASTERN.localize(datetime.combine(start_day, time(4, 0)))
        start = start_et.astimezone(timezone.utc)
        return start, end

    async def _process_single_symbol(
        self, symbol: str, as_of: datetime, sem: asyncio.Semaphore
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
                start, end = self._calculate_fetch_window(as_of)

                # Use Fetcher
                bars_data, fetch_metrics = await self.fetcher.fetch_bars(
                    sym, start, end, "1Min"
                )

                # Merge fetch metrics
                for k, v in fetch_metrics.items():
                    local[k] = int(local.get(k, 0)) + int(v)

                if not bars_data:
                    self.logger.warning("No bars found for technicals", symbol=sym)
                    return sym, None, local

                try:
                    tech_result = self.calculator.compute_technicals(bars_data)
                    if not tech_result:
                        local["technicals_fail"] += 1
                        return sym, None, local

                    (
                        price,
                        volume,
                        timestamp,
                        atr_pct,
                        intraday_range_pct,
                        gap_pct,
                        ema20_slope,
                        distance_from_vwap,
                        adx,
                        distance_from_ema20,
                        prior_day_high,
                        prior_day_low,
                        bb_upper,
                        bb_lower,
                        price_zscore,
                        premarket_vol,
                    ) = tech_result

                    avg_daily_volume = await asyncio.to_thread(
                        self.fetcher.fetch_avg_daily_volume,
                        sym,
                        end,
                        self.daily_volume_lookback_days,
                    )

                    p_high, p_low, p_close = await asyncio.to_thread(
                        self.fetcher.fetch_prior_day_stats,
                        sym,
                        end,
                    )
                    # Fix floating point equality check
                    if abs(prior_day_high) < 1e-9 or abs(prior_day_low) < 1e-9:
                        if p_high > 0:
                            prior_day_high = p_high
                            prior_day_low = p_low

                    session_open = self.calculator.calculate_session_open_price(
                        bars_data if isinstance(bars_data, list) else [], end
                    )
                    if p_close > 0 and session_open > 0:
                        gap_pct = (session_open - p_close) / p_close

                except Exception as e:
                    local["technicals_fail"] += 1
                    self.logger.error(
                        "Failed to compute technicals", symbol=sym, error=str(e)
                    )
                    return sym, None, local

                # Create features with NEUTRAL flow data
                feat = SymbolFeatures(
                    symbol=sym,
                    last_updated=(
                        timestamp
                        if isinstance(timestamp, datetime)
                        else datetime.fromisoformat(
                            str(timestamp).replace("Z", UTC_ZERO_STR)
                        )
                    ),
                    price=price,
                    avg_volume=(
                        float(avg_daily_volume)
                        if avg_daily_volume is not None
                        else float(volume)
                    ),
                    atr_pct=atr_pct,
                    intraday_range_pct=intraday_range_pct,
                    gap_pct=gap_pct,
                    ema20_slope=ema20_slope,
                    ema_trend_strength=abs(ema20_slope),
                    distance_from_vwap=distance_from_vwap,
                    premarket_volume=premarket_vol,
                    adx=adx,
                    distance_from_ema20=distance_from_ema20,
                    prior_day_high=prior_day_high,
                    prior_day_low=prior_day_low,
                    bb_upper=bb_upper,
                    bb_lower=bb_lower,
                    price_zscore=price_zscore,
                    # Neutral flow details
                    flow_zscore=0.0,
                    call_put_ratio=0.0,
                    large_sweeps_count=0,
                    aggressive_flow_share=0.0,
                    extra={
                        "flow_raw_count": 0,
                        "flow_bias": 0.0,
                        "volatility": atr_pct,
                        "last_bar_volume": float(volume),
                        "avg_daily_volume_days": int(self.daily_volume_lookback_days),
                    },
                )

                local["features_ok"] += 1
                return sym, feat, local

            except Exception as e:
                self.logger.error(
                    "Failed to compute technicals (outer)", symbol=sym, error=str(e)
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
            raise ValueError(
                "FeaturePipeline.compute_technicals_only requires as_of for deterministic behavior"
            )

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

        # Use symbols list directly
        results = await asyncio.gather(
            *[self._process_single_symbol(s, as_of, sem) for s in symbols]
        )
        for sym, feat, local in results:
            for k, v in local.items():
                metrics[k] = int(metrics.get(k, 0)) + int(v)
            if feat is not None:
                features[sym] = feat

        # Store intermediate metrics? Or merge later.
        # We'll merge in the wrapper.
        self.last_run_metrics = dict(metrics)  # partial update
        return features

    async def append_flow_features(
        self, features_map: Dict[str, SymbolFeatures]
    ) -> Dict[str, SymbolFeatures]:
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
            self.logger.info(
                "Unusual Whales flow integration disabled; skipping fetch."
            )
            return features_map

        async def _enrich_one(feat: SymbolFeatures) -> SymbolFeatures:
            sym = feat.symbol
            local_fail = 0

            async with sem_flow:
                # Add delay to respect rate limits (approx 2 req/sec safe limit?)
                await asyncio.sleep(0.5)
                try:
                    # We need a date for the flow fetch. feature.last_updated should be consistent with as_of.
                    date_str = feat.last_updated.strftime("%Y-%m-%d")
                    flow_data = await self.fetcher.fetch_flow(sym, date_str)
                except Exception:
                    local_fail = 1
                    # Log already handled in fetcher but we can log context here too if needed
                    flow_data = []

            (c_p_ratio, f_zscore, sw_count, agg_share, f_bias) = (
                self.calculator.compute_flow_metrics(flow_data)
            )

            # Update feature object (dataclass is mutable-ish or we allow direct field update)
            # Python dataclasses are mutable by default.
            feat.flow_zscore = f_zscore
            feat.call_put_ratio = c_p_ratio
            feat.large_sweeps_count = sw_count
            feat.aggressive_flow_share = agg_share
            if feat.extra is None:
                feat.extra = {}
            feat.extra["flow_raw_count"] = len(flow_data) if flow_data else 0
            feat.extra["flow_bias"] = f_bias

            if local_fail:
                metrics["uw_fetch_fail"] = int(metrics.get("uw_fetch_fail", 0)) + 1

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

    async def compute_features(
        self, symbols: List[str], as_of: Optional[datetime] = None
    ) -> Dict[str, SymbolFeatures]:
        """
        Computes features for a list of symbols (Full Pipeline).
        """
        # Run Stage 1
        features = await self.compute_technicals_only(symbols, as_of=as_of)

        # Run Stage 2 (on ALL symbols, preserving legacy behavior)
        # Using a copy of keys to pass list is implied by append_flow_features taking the dict
        await self.append_flow_features(features)

        # Finalize metrics
        if hasattr(self, "last_run_metrics"):
            self.logger.info("FeaturePipeline summary", **self.last_run_metrics)

        return features
