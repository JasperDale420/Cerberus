"""V8a: Trend Pullback with Inside Day Pattern.

Buy when price is in an uptrend (above SMA40), pulls back near SMA20,
and forms an inside day (compression before expansion). Structurally
different from mean reversion — this is trend continuation.
Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v8a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        # Trend filter
        self.trend_sma_period = int(config.get("trend_sma_period", 40))
        self.pullback_sma_period = int(config.get("pullback_sma_period", 20))
        self.pullback_pct = float(config.get("pullback_pct", 0.03))
        # Inside day
        self.inside_day_required = bool(config.get("inside_day_required", True))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 50))
        self.drawdown_max = float(config.get("drawdown_max", 0.12))
        # Volume filter
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_mult = float(config.get("vol_min_mult", 0.5))
        # ATR for stop/target
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 7))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _atr(bars: list[Bar], period: int) -> Optional[float]:
        if len(bars) < period + 1:
            return None
        trs = []
        for i in range(-period, 0):
            b = bars[i]
            prev_close = bars[i - 1].close
            tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
            trs.append(tr)
        return sum(trs) / period

    @staticmethod
    def _is_inside_day(bars: list[Bar]) -> bool:
        """Check if the latest bar is an inside day (range within previous bar)."""
        if len(bars) < 2:
            return False
        curr = bars[-1]
        prev = bars[-2]
        return curr.high <= prev.high and curr.low >= prev.low

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # --- Filter 1: Trend — price above SMA40 (uptrend) ---
        trend_sma = self._sma(closes, self.trend_sma_period)
        if trend_sma is None or bar.close < trend_sma:
            return None

        # --- Filter 2: Pullback — price near SMA20 ---
        pullback_sma = self._sma(closes, self.pullback_sma_period)
        if pullback_sma is None:
            return None
        # Price should be within pullback_pct of the pullback SMA (near or slightly below)
        distance = (bar.close - pullback_sma) / pullback_sma
        if distance > self.pullback_pct:
            return None  # Too far above — not a pullback
        if distance < -self.pullback_pct:
            return None  # Too far below — trend might be breaking

        # --- Filter 3: SMA alignment (fast > slow = healthy trend) ---
        if pullback_sma <= trend_sma:
            return None

        # --- Filter 4: Inside day (compression) ---
        if self.inside_day_required and not self._is_inside_day(bars):
            return None

        # --- Filter 5: Drawdown guard ---
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.drawdown_max:
            return None

        # --- Filter 6: Regime filter ---
        labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = labels.get("regime_trend", "FLAT")
        regime_vol = labels.get("regime_vol", "NORMAL")
        if regime_trend == "DOWN":
            return None
        if regime_vol in ("HIGH", "SHOCK"):
            return None

        # --- Filter 7: Volume filter ---
        volumes = [b.volume for b in bars]
        if len(volumes) >= self.vol_avg_period:
            avg_vol = sum(volumes[-self.vol_avg_period :]) / self.vol_avg_period
            if avg_vol > 0 and bar.volume < avg_vol * self.vol_min_mult:
                return None

        # --- ATR for stop/target ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = bar.close + self.target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "pullback_dist": round(distance, 4),
                "trend_sma": round(trend_sma, 2),
                "inside_day": self._is_inside_day(bars),
                "atr": round(atr, 4),
                "regime": f"{regime_trend}+{regime_vol}",
            },
        )
