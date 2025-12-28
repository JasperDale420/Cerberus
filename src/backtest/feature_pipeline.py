from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pytz  # type: ignore

from src.core.domain import Bar, SymbolFeatures
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

        fp_cfg = (
            (self.config.get("feature_pipeline") or {})
            if isinstance(self.config, dict)
            else {}
        )
        self.daily_volume_lookback_days = int(
            fp_cfg.get("daily_volume_lookback_days", 20)
        )

        self._series: Dict[str, _Series] = {}
        self._daily_volumes: Dict[str, Dict[Any, float]] = {}
        self._build_index(bars_by_symbol)

        self.last_run_metrics: Dict[str, int] = {}

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

        end_et = end.astimezone(US_EASTERN)
        start_day = end_et.date()
        if end_et.time() < time(4, 0):
            start_day = start_day - timedelta(days=1)
        start_et = US_EASTERN.localize(datetime.combine(start_day, time(4, 0)))
        start = start_et.astimezone(timezone.utc)
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

    async def compute_technicals_only(
        self, symbols: List[str], as_of: Optional[datetime] = None
    ) -> Dict[str, SymbolFeatures]:
        if as_of is None:
            raise ValueError(
                "BacktestFeaturePipeline.compute_technicals_only requires as_of for deterministic behavior"
            )
        now = as_of
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        metrics: Dict[str, int] = {
            "symbols_total": int(len(symbols)),
            "features_ok": 0,
            "no_bars": 0,
            "technicals_fail": 0,
        }

        start, end = self._calculate_fetch_window(now)
        out: Dict[str, SymbolFeatures] = {}

        for s in symbols:
            sym = str(s).strip().upper()
            bars = self._slice_bars(sym, start, end)
            if not bars:
                metrics["no_bars"] += 1
                continue
            tech = self.calculator.compute_technicals(bars, current_time=now)
            if not tech:
                metrics["technicals_fail"] += 1
                continue

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
            ) = tech

            avg_daily_volume = self._avg_daily_volume(sym, now)
            feat = SymbolFeatures(
                symbol=sym,
                last_updated=timestamp if isinstance(timestamp, datetime) else now,
                price=float(price),
                avg_volume=float(
                    avg_daily_volume if avg_daily_volume is not None else volume
                ),
                atr_pct=float(atr_pct),
                intraday_range_pct=float(intraday_range_pct),
                gap_pct=float(gap_pct),
                ema20_slope=float(ema20_slope),
                ema_trend_strength=float(abs(float(ema20_slope))),
                distance_from_vwap=float(distance_from_vwap),
                premarket_volume=float(premarket_vol),
                adx=float(adx),
                distance_from_ema20=float(distance_from_ema20),
                prior_day_high=float(prior_day_high),
                prior_day_low=float(prior_day_low),
                bb_upper=float(bb_upper),
                bb_lower=float(bb_lower),
                price_zscore=float(price_zscore),
                flow_zscore=0.0,
                call_put_ratio=0.0,
                large_sweeps_count=0,
                aggressive_flow_share=0.0,
                extra={
                    "flow_raw_count": 0,
                    "flow_bias": 0.0,
                    "volatility": float(atr_pct),
                    "last_bar_volume": float(volume),
                    "avg_daily_volume_days": int(self.daily_volume_lookback_days),
                },
            )
            out[sym] = feat
            metrics["features_ok"] += 1

        self.last_run_metrics = dict(metrics)
        return out

    async def append_flow_features(
        self, features_map: Dict[str, SymbolFeatures]
    ) -> Dict[str, SymbolFeatures]:
        # Deterministic backtests do not fetch external flow by default.
        return features_map
