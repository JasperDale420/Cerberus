"""Breakout-Pullback with Breadth Filter (evolved from vol_breakout seed).

Buy pullbacks after recent breakout. Requires breadth confirmation
(multiple up days in lookback) to naturally avoid downtrending stocks.
Regime-adaptive stops for consistency.
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
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.0))
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

        if len(bars) < 25:
            return None

        # Recent breakout — 20-day high within last 5 bars
        high_20d = max(b.high for b in bars[-21:-1])
        recent_highs = [b.high for b in bars[-6:-1]]
        had_breakout = any(h >= high_20d * 0.99 for h in recent_highs)
        if not had_breakout:
            return None

        # Pullback — today near but below recent high (0.5%-5%)
        recent_high = max(recent_highs)
        pullback_pct = (recent_high - bar.close) / recent_high
        if pullback_pct < 0.005 or pullback_pct > 0.05:
            return None

        # Close in upper half of today's range
        today_range = bar.high - bar.low
        if today_range < 1e-9:
            return None
        close_position = (bar.close - bar.low) / today_range
        if close_position < 0.5:
            return None

        # Trend: above SMA(20)
        sma20 = self._sma(closes, 20)
        if sma20 is None or bar.close <= sma20:
            return None

        # Breadth: at least 5 of last 10 bars were up days
        if len(bars) < 11:
            return None
        up_days = sum(1 for i in range(-10, 0) if bars[i].close > bars[i - 1].close)
        if up_days < 5:
            return None

        # Regime-adaptive stops
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
                "up_days_10": up_days,
                "seed": "vol_breakout",
            },
        )
