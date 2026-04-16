"""Range Expansion with regime-adaptive filters (evolved from vol_breakout seed).

Buy wide-range bars with strong closes. Skip DOWN trend entirely.
Tighter risk management for consistency.
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
        self.min_bars = int(config.get("min_bars", 25))
        self.atr_period = int(config.get("atr_period", 14))
        self.range_expansion_mult = float(config.get("range_expansion_mult", 1.2))
        self.close_strength_min = float(config.get("close_strength_min", 0.65))
        self.momentum_lookback = int(config.get("momentum_lookback", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 3))

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
        # Skip DOWN trend — breakout strategies lose in downtrends
        if regime_labels.get("regime_trend") == "DOWN":
            return None

        bars = list(symbol_state.bars)

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Range expansion: today's range > 1.2x ATR
        today_range = bar.high - bar.low
        if today_range < 1e-9 or today_range < self.range_expansion_mult * atr:
            return None

        # Close strength: close in upper portion of today's range
        close_position = (bar.close - bar.low) / today_range
        if close_position < self.close_strength_min:
            return None

        # Momentum: close above N days ago (short-term upward movement)
        if len(bars) < self.momentum_lookback + 1:
            return None
        past_close = bars[-(self.momentum_lookback + 1)].close
        if bar.close <= past_close:
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
                "atr": round(atr, 4),
                "range_expansion": round(today_range / atr, 2),
                "close_strength": round(close_position, 2),
                "momentum": round((bar.close - past_close) / past_close * 100, 2),
                "seed": "vol_breakout",
            },
        )
