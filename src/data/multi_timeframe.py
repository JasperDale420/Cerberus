from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from src.core import time_utils
from src.core.domain import Bar, OrderSide, SymbolState
from src.core.indicators import (
    RollingADX,
    RollingATR,
    RollingEMA,
    RollingRSI,
    RollingStd,
)


@dataclass
class _TFCache:
    """Internal cache for a single timeframe's computed indicators."""

    bar_count: int = 0
    last_bar_time: datetime | None = None
    ema: dict[int, float | None] = field(default_factory=dict)
    rsi: dict[int, float | None] = field(default_factory=dict)
    atr: dict[int, float | None] = field(default_factory=dict)
    adx: float | None = None
    bb: dict[int, tuple[float, float]] = field(default_factory=dict)  # period -> (mean, std)
    vwap_distance: float | None = None
    swing_highs: list[float] = field(default_factory=list)
    swing_lows: list[float] = field(default_factory=list)


# Timeframe weights: higher timeframe = more weight in alignment scores.
_TF_WEIGHTS: dict[str, float] = {"1m": 0.20, "5m": 0.35, "15m": 0.45}


class MultiTimeframeAnalyzer:
    """Provides multi-timeframe indicator access and alignment scoring.

    Reads bar deques from a SymbolState (bars_1m, bars_5m, bars_15m),
    computes Rolling* indicators on each timeframe, caches results, and
    exposes alignment scores that summarise cross-timeframe agreement.
    """

    TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m")

    def __init__(self, symbol_state: SymbolState) -> None:
        self._state = symbol_state
        self._cache: dict[str, _TFCache] = {tf: _TFCache() for tf in self.TIMEFRAMES}

    # ------------------------------------------------------------------
    # Public: alignment scores
    # ------------------------------------------------------------------

    def get_trend_alignment(self, side: OrderSide) -> float:
        """Score 0.0-1.0 measuring how many timeframes agree with proposed direction.

        For BUY: each TF where EMA20 > EMA50 adds to score.
        For SELL: each TF where EMA20 < EMA50 adds to score.

        Weighting: 1m = 0.2, 5m = 0.35, 15m = 0.45.
        Returns 0.0 if no data available.
        """
        score = 0.0
        for tf in self.TIMEFRAMES:
            ema20 = self.get_ema(tf, 20)
            ema50 = self.get_ema(tf, 50)
            if ema20 is None or ema50 is None:
                continue
            if (side == OrderSide.BUY and ema20 > ema50) or (side == OrderSide.SELL and ema20 < ema50):
                score += _TF_WEIGHTS[tf]
        return score

    def get_mean_reversion_alignment(self) -> float:
        """Score 0.0-1.0 measuring how many timeframes show flat/choppy conditions.

        Flat = EMA20 within 0.15% of EMA50 on that timeframe.
        Higher score = better for mean reversion trades.

        Weighting: 1m = 0.2, 5m = 0.35, 15m = 0.45.
        """
        score = 0.0
        for tf in self.TIMEFRAMES:
            ema20 = self.get_ema(tf, 20)
            ema50 = self.get_ema(tf, 50)
            if ema20 is None or ema50 is None:
                continue
            if abs(ema50) < 1e-9:
                continue
            spread_pct = abs(ema20 - ema50) / abs(ema50)
            if spread_pct < 0.0015:  # 0.15%
                score += _TF_WEIGHTS[tf]
        return score

    # ------------------------------------------------------------------
    # Public: per-timeframe indicator getters
    # ------------------------------------------------------------------

    def get_ema(self, timeframe: str, period: int) -> float | None:
        """Get EMA value for a timeframe. Returns None if insufficient data."""
        self._ensure_cache(timeframe)
        cache = self._cache.get(timeframe)
        if cache is None:
            return None
        return cache.ema.get(period)

    def get_rsi(self, timeframe: str, period: int = 14) -> float | None:
        """Get RSI value for a timeframe."""
        self._ensure_cache(timeframe)
        cache = self._cache.get(timeframe)
        if cache is None:
            return None
        return cache.rsi.get(period)

    def get_atr(self, timeframe: str, period: int = 14) -> float | None:
        """Get ATR value for a timeframe."""
        self._ensure_cache(timeframe)
        cache = self._cache.get(timeframe)
        if cache is None:
            return None
        return cache.atr.get(period)

    def get_adx(self, timeframe: str) -> float | None:
        """Get ADX value for a timeframe."""
        self._ensure_cache(timeframe)
        cache = self._cache.get(timeframe)
        if cache is None:
            return None
        return cache.adx

    def get_vwap_distance(self, timeframe: str) -> float | None:
        """Get distance from VWAP as a percentage for a timeframe.

        Calculated as ``(close - vwap) / vwap``.
        """
        self._ensure_cache(timeframe)
        cache = self._cache.get(timeframe)
        if cache is None:
            return None
        return cache.vwap_distance

    def get_bb_position(self, timeframe: str, period: int = 20, std_mult: float = 2.0) -> float | None:
        """Get position within Bollinger Bands as -1.0 to 1.0.

        -1.0 = at/below lower band
         0.0 = at middle (SMA)
         1.0 = at/above upper band
        """
        self._ensure_cache(timeframe)
        cache = self._cache.get(timeframe)
        if cache is None:
            return None
        bb_data = cache.bb.get(period)
        if bb_data is None:
            return None
        mean, std = bb_data
        if std <= 0.0:
            return None
        bars = self._bars_for(timeframe)
        if not bars:
            return None
        close = float(bars[-1].close)
        half_width = std * std_mult
        if half_width <= 0.0:
            return None
        # Map close into [-1, 1] relative to the band.
        raw = (close - mean) / half_width
        return max(-1.0, min(1.0, raw))

    def get_swing_highs(self, timeframe: str, lookback: int = 5) -> list[float]:
        """Get recent swing high prices (local maxima) sorted by recency."""
        self._ensure_cache(timeframe)
        bars = self._bars_for(timeframe)
        if len(bars) < (2 * lookback + 1):
            return []
        return self._detect_swings(bars, lookback, mode="high")

    def get_swing_lows(self, timeframe: str, lookback: int = 5) -> list[float]:
        """Get recent swing low prices (local minima) sorted by recency."""
        self._ensure_cache(timeframe)
        bars = self._bars_for(timeframe)
        if len(bars) < (2 * lookback + 1):
            return []
        return self._detect_swings(bars, lookback, mode="low")

    # ------------------------------------------------------------------
    # Internal: bar access helpers
    # ------------------------------------------------------------------

    def _bars_for(self, timeframe: str) -> deque[Bar]:
        """Return the bar deque for a given timeframe label."""
        if timeframe == "1m":
            return self._state.bars_1m
        if timeframe == "5m":
            return self._state.bars_5m
        if timeframe == "15m":
            return self._state.bars_15m
        return deque()

    # ------------------------------------------------------------------
    # Internal: cache management
    # ------------------------------------------------------------------

    def _ensure_cache(self, timeframe: str) -> None:
        """Recompute indicators for *timeframe* if new bars have arrived."""
        bars = self._bars_for(timeframe)
        if not bars:
            return

        cache = self._cache.get(timeframe)
        if cache is None:
            cache = _TFCache()
            self._cache[timeframe] = cache

        n = len(bars)
        last_time = bars[-1].time
        if cache.bar_count == n and cache.last_bar_time == last_time:
            return  # cache still fresh

        self._compute_tf(timeframe, bars, cache)
        cache.bar_count = n
        cache.last_bar_time = last_time

    def _compute_tf(self, timeframe: str, bars: deque[Bar], cache: _TFCache) -> None:
        """Indicator recomputation for one timeframe from its bar deque.

        Fast path: if pre-computed numpy arrays are installed (backtest mode),
        look up the current bar index and read values directly — O(1).
        Slow path: replay all bars through Rolling* Python objects.
        """
        # --- Fast path: pre-computed arrays (backtest acceleration) ---
        try:
            from src.backtest.indicator_precompute import get_precomputed

            precomp = get_precomputed(self._state.symbol, timeframe)
        except ImportError:
            precomp = None

        if precomp and len(precomp) > 0:
            n = len(bars)
            if timeframe == "1m":
                # 1m arrays are indexed directly by bar count (same length as bars_1m)
                idx = n - 1
                if idx < 0:
                    return
                for p in (9, 20, 50, 200):
                    arr = precomp.get(f"ema_{p}")
                    if arr is not None and idx < len(arr):
                        v = float(arr[idx])
                        cache.ema[p] = None if v != v else v  # NaN check
                for p in (2, 14):
                    arr = precomp.get(f"rsi_{p}")
                    if arr is not None and idx < len(arr):
                        v = float(arr[idx])
                        cache.rsi[p] = None if v != v else v
                arr = precomp.get("atr_14")
                if arr is not None and idx < len(arr):
                    v = float(arr[idx])
                    cache.atr[14] = None if v != v else v
                arr = precomp.get("adx_14")
                if arr is not None and idx < len(arr):
                    v = float(arr[idx])
                    cache.adx = None if v != v else v
                arr_m = precomp.get("bb_mean_20")
                arr_s = precomp.get("bb_std_20")
                if arr_m is not None and arr_s is not None and idx < len(arr_m):
                    cache.bb[20] = (float(arr_m[idx]), float(arr_s[idx]))
                arr = precomp.get("vwap_distance")
                if arr is not None and idx < len(arr):
                    v = float(arr[idx])
                    cache.vwap_distance = None if v != v else v
                return
            else:
                # 5m/15m: use the _1m_index mapping
                idx_map = precomp.get("_1m_index")
                if idx_map is not None:
                    # The 1m bar count corresponds to bars_1m length
                    bars_1m = self._state.bars_1m
                    n_1m = len(bars_1m)
                    if n_1m > 0 and (n_1m - 1) < len(idx_map):
                        agg_idx = int(idx_map[n_1m - 1])
                        if agg_idx < 0:
                            return  # No completed higher-TF bar yet
                        for p in (9, 20, 50, 200):
                            arr = precomp.get(f"ema_{p}")
                            if arr is not None and agg_idx < len(arr):
                                v = float(arr[agg_idx])
                                cache.ema[p] = None if v != v else v
                        for p in (2, 14):
                            arr = precomp.get(f"rsi_{p}")
                            if arr is not None and agg_idx < len(arr):
                                v = float(arr[agg_idx])
                                cache.rsi[p] = None if v != v else v
                        arr = precomp.get("atr_14")
                        if arr is not None and agg_idx < len(arr):
                            v = float(arr[agg_idx])
                            cache.atr[14] = None if v != v else v
                        arr = precomp.get("adx_14")
                        if arr is not None and agg_idx < len(arr):
                            v = float(arr[agg_idx])
                            cache.adx = None if v != v else v
                        arr_m = precomp.get("bb_mean_20")
                        arr_s = precomp.get("bb_std_20")
                        if arr_m is not None and arr_s is not None and agg_idx < len(arr_m):
                            cache.bb[20] = (float(arr_m[agg_idx]), float(arr_s[agg_idx]))
                        arr = precomp.get("vwap_distance")
                        if arr is not None and agg_idx < len(arr):
                            v = float(arr[agg_idx])
                            cache.vwap_distance = None if v != v else v
                        return

        # --- Slow path: replay all bars through Rolling* objects ---
        ema_periods = (9, 20, 50, 200)
        rsi_periods = (2, 14)
        atr_period = 14
        adx_period = 14
        bb_period = 20

        ema_objs: dict[int, RollingEMA] = {p: RollingEMA.from_period(p) for p in ema_periods}
        rsi_objs: dict[int, RollingRSI] = {p: RollingRSI(period=p) for p in rsi_periods}
        atr_obj = RollingATR(period=atr_period)
        adx_obj = RollingADX(period=adx_period)
        bb_obj = RollingStd.create(bb_period)

        cum_pv = 0.0
        cum_vol = 0.0
        current_date = None

        for bar in bars:
            close = float(bar.close)
            high = float(bar.high)
            low = float(bar.low)
            volume = float(bar.volume)

            for _p, obj in ema_objs.items():
                obj.update(close)
            for _p, obj in rsi_objs.items():
                obj.update(close)
            atr_obj.update(high, low, close)
            adx_obj.update(high, low, close)
            bb_obj.update(close)

            # Reset VWAP accumulators at each new trading day (ET date boundary).
            bar_date = time_utils.to_eastern_time(bar.time).date()
            if bar_date != current_date:
                cum_pv = 0.0
                cum_vol = 0.0
                current_date = bar_date

            if bar.vwap is not None and volume > 0:
                cum_pv += float(bar.vwap) * volume
            elif volume > 0:
                tp = (high + low + close) / 3.0
                cum_pv += tp * volume
            cum_vol += volume

        for p, obj in ema_objs.items():
            cache.ema[p] = float(obj.value) if obj.value is not None else None

        for p, obj in rsi_objs.items():
            cache.rsi[p] = float(obj.value) if obj.value is not None else None

        cache.atr[atr_period] = float(atr_obj.value) if atr_obj.value is not None else None
        cache.adx = float(adx_obj.value) if adx_obj.value is not None else None

        n_bb = len(bb_obj.values)
        if n_bb > 0:
            bb_mean = bb_obj.sum / n_bb
            bb_var = max(0.0, (bb_obj.sumsq / n_bb) - (bb_mean * bb_mean))
            cache.bb[bb_period] = (bb_mean, bb_var**0.5)
        else:
            cache.bb[bb_period] = (0.0, 0.0)

        if cum_vol > 0 and len(bars) > 0:
            vwap = cum_pv / cum_vol
            last_close = float(bars[-1].close)
            cache.vwap_distance = (last_close - vwap) / vwap if vwap > 0 else None
        else:
            cache.vwap_distance = None

    # ------------------------------------------------------------------
    # Internal: swing detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_swings(bars: deque[Bar], lookback: int, mode: str) -> list[float]:
        """Detect swing highs or lows from a bar deque (backward-only).

        A swing high at index *i* means ``bar[i].high`` is >= the high of
        the preceding *lookback* bars.  Swing lows work analogously with
        ``bar[i].low``.  Only past bars (indices < i) are inspected so that
        no future data is used.

        Returns prices sorted most-recent first.
        """
        n = len(bars)
        results: list[tuple[int, float]] = []

        for i in range(lookback, n):
            if mode == "high":
                val = float(bars[i].high)
                is_swing = all(val >= float(bars[i - j].high) for j in range(1, lookback + 1))
            else:
                val = float(bars[i].low)
                is_swing = all(val <= float(bars[i - j].low) for j in range(1, lookback + 1))
            if is_swing:
                results.append((i, val))

        # Most recent first.
        results.sort(key=lambda t: t[0], reverse=True)
        return [price for _, price in results]
