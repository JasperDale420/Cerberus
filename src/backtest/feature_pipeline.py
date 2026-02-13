from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pytz  # type: ignore

from src.core.domain import Bar, Regime, ScanResult, SymbolFeatures, WatchlistSymbol
from src.core.logger import StructuredLogger
from src.data.calculator import FeatureCalculator

US_EASTERN = pytz.timezone("US/Eastern")


@dataclass(frozen=True)
class _Series:
    times: List[datetime]
    bars: List[Bar]


class BacktestFeaturePipeline:
    """
    Offline feature pipeline used to replay `Scanner` deterministically during backtests.

    Uses already-loaded bars (from BacktestRunner) to compute `SymbolFeatures` at a scan_time.
    """

    async def scan(self, regime: Regime, scan_time: datetime) -> ScanResult:
        """
        Shim for the Scanner protocol used during backtests.
        Wraps compute_technicals_only results into a ScanResult.
        """
        symbols = list(self._series.keys())
        features_map = await self.compute_technicals_only(symbols, as_of=scan_time)

        # Build WatchlistSymbol objects expected by ExecutionEngine
        watchlist = []
        routing_cfg = self.config.get("strategy_routing", {})
        regime_key = regime.value if hasattr(regime, "value") else str(regime)
        regime_strategies = list(routing_cfg.get(regime_key, []))

        watchlist = [
            WatchlistSymbol(
                symbol=sym,
                score=float(feat.relative_strength),
                strategies=regime_strategies,
                features=feat,
            )
            for sym, feat in features_map.items()
        ]

        return ScanResult(
            generated_at=scan_time,
            regime=regime,
            watchlist=watchlist,
        )

    def __init__(
        self,
        bars_by_symbol: Dict[str, List[Bar]],
        logger: StructuredLogger,
        config: Optional[Dict[str, Any]] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.logger = logger
        self.config = config or {}
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.calculator = FeatureCalculator()

        # Shim for PairScanner/Scanner which expects pipeline.fetcher.fetch_bars
        self.fetcher = self

        fp_cfg = (self.config.get("feature_pipeline") or {}) if isinstance(self.config, dict) else {}
        self.daily_volume_lookback_days = int(fp_cfg.get("daily_volume_lookback_days", 20))

        self._series: Dict[str, _Series] = {}
        self._daily_volumes: Dict[str, Dict[Any, float]] = {}
        self._build_index(bars_by_symbol)

        self.last_run_metrics: Dict[str, int] = {}

        # Per-day feature cache: Alpha Ranking factors (50/200-day SMAs) only change daily,
        # so we cache them to avoid re-slicing 200k+ bars per symbol on every intraday scan.
        self._daily_feature_cache: Dict[str, SymbolFeatures] = {}
        self._cache_date: Optional[date] = None

    def _build_index(self, bars_by_symbol: Dict[str, List[Bar]]) -> None:
        for sym_raw, bars in bars_by_symbol.items():
            sym = str(sym_raw).strip().upper()
            b_sorted = sorted(
                [b for b in bars if isinstance(b, Bar)],
                key=lambda b: (b.time, b.symbol),
            )
            times = []
            dvols: Dict[Any, float] = {}
            for b in b_sorted:
                t = b.time
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                    b.time = t
                times.append(t)
                d = t.astimezone(US_EASTERN).date()
                dvols[d] = float(dvols.get(d, 0.0)) + float(b.volume)
            self._series[sym] = _Series(times=times, bars=b_sorted)
            self._daily_volumes[sym] = dvols

    def _calculate_fetch_window(self, as_of: datetime) -> tuple[datetime, datetime]:
        end = as_of
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        # Expand lookback to 365 days (~252 trading days) to ensure we have sufficient
        # historical data for 200-day SMAs (Alpha Ranking Factors).
        start = end - timedelta(days=365)
        return start, end

    def _slice_bars(self, sym: str, start: datetime, end: datetime) -> List[Bar]:
        series = self._series.get(sym)
        if series is None:
            return []
        i0 = bisect.bisect_left(series.times, start)
        i1 = bisect.bisect_right(series.times, end)
        return series.bars[i0:i1]

    def _avg_daily_volume(self, sym: str, as_of: datetime) -> Optional[float]:
        vols = self._daily_volumes.get(sym) or {}
        if not vols:
            return None
        as_of_et = as_of.astimezone(US_EASTERN)
        day = as_of_et.date()
        prior_days = sorted([d for d in vols.keys() if d < day])
        lookback = max(1, int(self.daily_volume_lookback_days))
        if prior_days:
            use = prior_days[-lookback:]
            return float(sum(float(vols[d]) for d in use) / float(len(use)))
        # Fallback: use volume accumulated today (partial day).
        return float(vols.get(day, 0.0))

    async def fetch_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min"
    ) -> tuple[List[Bar], Dict[str, Any]]:
        """
        Shim for DataFetcher.fetch_bars used by Scanner._scan_pairs.
        """
        # Note: Backtest currently ignores timeframe and uses whatever bars were provided
        bars = self._slice_bars(symbol, start, end)
        return bars, {}

    async def compute_technicals_only(
        self, symbols: List[str], as_of: Optional[datetime] = None
    ) -> Dict[str, SymbolFeatures]:
        # Complies with Scanner interface which is async
        if as_of is None:
            raise ValueError(
                "BacktestFeaturePipeline.compute_technicals_only requires as_of for deterministic behavior"
            )
        now = as_of
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # Cache invalidation: clear cache at day boundary
        current_date = now.date()
        if self._cache_date is None or self._cache_date != current_date:
            self._daily_feature_cache.clear()
            self._cache_date = current_date
            self.logger.info("Feature cache cleared for new trading day", date=str(current_date))

        metrics: Dict[str, int] = {
            "symbols_total": int(len(symbols)),
            "features_ok": 0,
            "no_bars": 0,
            "technicals_fail": 0,
        }

        start, end = self._calculate_fetch_window(now)
        out: Dict[str, SymbolFeatures] = {}

        # 1. Fetch benchmark (SPY) for Relative Strength
        index_sym = str(self.config.get("index_symbol", "SPY")).upper()
        benchmark_bars = []
        if index_sym in self._series:
            benchmark_bars = self._slice_bars(index_sym, start, end)

        for s in symbols:
            sym = str(s).strip().upper()

            # Check cache first (alpha factors only change daily, not intraday)
            if sym in self._daily_feature_cache:
                out[sym] = self._daily_feature_cache[sym]
                metrics["features_ok"] += 1
                continue

            bars = self._slice_bars(sym, start, end)
            if not bars:
                metrics["no_bars"] += 1
                continue
            tech = self.calculator.compute_technicals(bars)
            if not tech:
                metrics["technicals_fail"] += 1
                continue

            # 2. Compute Relative Strength vs SPY
            # Anchor RS to 9:30 AM ET (Market Open) for institutional baseline
            market_open = US_EASTERN.localize(datetime.combine(now.date(), time(9, 30)))
            if bars and benchmark_bars:
                common_start = max(market_open, bars[0].time, benchmark_bars[0].time)
                sym_al = [b.close for b in bars if b.time >= common_start]
                ben_al = [b.close for b in benchmark_bars if b.time >= common_start]
                rs = self.calculator.calculate_relative_strength(sym_al, ben_al)
            else:
                rs = 0.0

            avg_daily_volume = self._avg_daily_volume(sym, now)
            feat = SymbolFeatures(
                symbol=sym,
                last_updated=(tech.timestamp if isinstance(tech.timestamp, datetime) else now),
                price=float(tech.price),
                avg_volume=float(avg_daily_volume if avg_daily_volume is not None else tech.volume),
                atr_pct=float(tech.atr_pct),
                intraday_range_pct=float(tech.intraday_range_pct),
                gap_pct=float(tech.gap_pct),
                ema20_slope=float(tech.ema20_slope),
                ema_trend_strength=float(abs(float(tech.ema20_slope))),
                distance_from_vwap=float(tech.distance_from_vwap),
                premarket_volume=float(tech.premarket_volume),
                adx=float(tech.adx),
                distance_from_ema20=float(tech.distance_from_ema20),
                prior_day_high=float(tech.prior_day_high),
                prior_day_low=float(tech.prior_day_low),
                atr=float(tech.atr),
                bb_upper=float(tech.bb_upper),
                bb_lower=float(tech.bb_lower),
                price_zscore=float(tech.price_zscore),
                # options flow (Mostly unobserved in technical-only backtest)
                flow_zscore=0.0,
                call_put_ratio=0.0,
                large_sweeps_count=0,
                aggressive_flow_share=0.0,
                # Fusion Signals
                relative_strength=float(rs),
                flow_bias=0.0,
                dof_score=0.0,
                orb_high=float(tech.orb_high),
                orb_low=float(tech.orb_low),
                # Alpha Factors for RankingEngine
                ma_dist_50=float(tech.ma_dist_50),
                ma_dist_200=float(tech.ma_dist_200),
                extra={
                    "flow_raw_count": 0,
                    "flow_bias": 0.0,
                    "volatility": float(tech.atr_pct),
                    "last_bar_volume": float(tech.volume),
                    "avg_daily_volume_days": int(self.daily_volume_lookback_days),
                    "atr": float(tech.atr),
                },
            )
            # Cache the computed features for subsequent intraday scans
            self._daily_feature_cache[sym] = feat
            out[sym] = feat
            metrics["features_ok"] += 1

        self.last_run_metrics = dict(metrics)
        return out

    async def append_flow_features(self, features_map: Dict[str, SymbolFeatures]) -> Dict[str, SymbolFeatures]:
        # Complies with Scanner interface which is async
        # Deterministic backtests do not fetch external flow by default.
        return features_map
