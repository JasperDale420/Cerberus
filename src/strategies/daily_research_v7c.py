"""Breakout-Pullback Strategy (evolved from vol_breakout seed).

Buy pullbacks to support after recent breakout (new 20-day high within last 5 days).
Better risk:reward than chasing breakouts. Regime-filtered for consistency.
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
        self.min_bars = int(config.get("min_bars", 55))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))

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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        atr = self._atr(bars, 14)
        if atr is None or atr < 1e-9:
            return None

        if len(bars) < 55:
            return None

        # Step 1: Recent breakout — a 20-day high was made within the last 5 bars
        high_20d = max(b.high for b in bars[-21:-1])
        recent_highs = [b.high for b in bars[-6:-1]]
        had_breakout = any(h >= high_20d * 0.99 for h in recent_highs)
        if not had_breakout:
            return None

        # Step 2: Pullback — today's close is below the recent high but not too far
        recent_high = max(recent_highs)
        pullback_pct = (recent_high - bar.close) / recent_high
        if pullback_pct < 0.005 or pullback_pct > 0.05:
            return None

        # Step 3: Close strength — close in upper half of today's range (not weak)
        today_range = bar.high - bar.low
        if today_range < 1e-9:
            return None
        close_position = (bar.close - bar.low) / today_range
        if close_position < 0.5:
            return None

        # Step 4: Trend — above both SMA(20) and SMA(50)
        sma20 = self._sma(closes, 20)
        sma50 = self._sma(closes, 50)
        if sma20 is None or sma50 is None:
            return None
        if bar.close <= sma20 or bar.close <= sma50:
            return None

        # Regime-adaptive stops: tighter in DOWN/HIGH
        regime_trend = regime_labels.get("regime_trend", "UP")
        regime_vol = regime_labels.get("regime_vol", "NORMAL")
        if regime_trend == "DOWN" or regime_vol == "HIGH":
            stop_mult = self.stop_atr_mult * 0.7
        else:
            stop_mult = self.stop_atr_mult

        stop = bar.close - stop_mult * atr
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
                "pullback_pct": round(pullback_pct * 100, 2),
                "close_pos": round(close_position, 2),
                "seed": "vol_breakout",
            },
        )
