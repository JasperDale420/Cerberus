"""Inside Day Breakout (vol_breakout archetype).

Buy after inside day (range contraction) resolves upward with volume.
Inside day = today's high < yesterday's high AND today's low > yesterday's low.
Next day breakout above inside day's high = entry signal.
Vol filter: require above-average ATR for context.
Skip DOWN regime, skip Fridays.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v7c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 30))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.atr_expansion = float(config.get("atr_expansion", 1.0))

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
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

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

        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings"):
            return None
        if regime_labels.get("regime_vol") == "SHOCK":
            return None
        if regime_labels.get("near_fomc"):
            return None

        # Skip Tuesdays, Thursdays, Fridays (historically negative)
        if hasattr(bar.time, "weekday") and bar.time.weekday() in (1, 3, 4):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        if len(bars) < 25:
            return None

        atr = self._atr(bars, 14)
        if atr is None or atr < 1e-9:
            return None

        prev_bar = bars[-2]
        prev_prev = bars[-3]

        # Inside day detection: previous bar had range inside the bar before it
        was_inside = prev_bar.high <= prev_prev.high and prev_bar.low >= prev_prev.low
        if not was_inside:
            return None

        # Quality: inside day range should be at most 85% of outer day range
        inner_range = prev_bar.high - prev_bar.low
        outer_range = prev_prev.high - prev_prev.low
        if outer_range < 1e-9 or inner_range / outer_range > 0.75:
            return None

        # Breakout: current bar closes above the inside day's high
        if bar.close <= prev_bar.high:
            return None

        # Close in upper half of today's range (strong breakout)
        today_range = bar.high - bar.low
        if today_range < 1e-9:
            return None
        close_pos = (bar.close - bar.low) / today_range
        if close_pos < 0.5:
            return None

        # Expansion: breakout day range should exceed inside day range
        if today_range <= inner_range:
            return None

        # Volume confirmation: above 60% of 20-day average (relaxed)
        volumes = [b.volume for b in bars]
        if len(volumes) < 20:
            return None
        avg_vol = sum(volumes[-20:]) / 20
        if avg_vol <= 0 or bar.volume < avg_vol * 0.8:
            return None

        # Trend context: above SMA(20) or close to it
        sma20 = self._sma(closes, 20)
        if sma20 is not None and bar.close < sma20 * 0.95:
            return None

        # Regime-adaptive stops
        regime_trend = regime_labels.get("regime_trend", "UP")
        regime_vol = regime_labels.get("regime_vol", "NORMAL")

        # More selective in DOWN: require stronger breakout
        if regime_trend == "DOWN":
            if close_pos < 0.7:
                return None
            stop_mult = self.stop_atr_mult * 0.8
            target_mult = self.target_atr_mult * 0.7
        elif regime_vol == "HIGH":
            stop_mult = self.stop_atr_mult * 0.9
            target_mult = self.target_atr_mult * 0.8
        else:
            stop_mult = self.stop_atr_mult
            target_mult = self.target_atr_mult

        stop = bar.close - stop_mult * atr
        target = bar.close + target_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "atr": round(atr, 4),
                "close_pos": round(close_pos, 2),
                "inside_day": True,
                "seed": "vol_breakout",
            },
        )
