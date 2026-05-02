"""
RegimeAdaptive Strategy

Type: adaptive
Description: Cross-regime generalist — EMA20 momentum with ATR stops.
  UP: buy when price pulls back to EMA20 from above
  DOWN: short when price bounces to EMA20 from below
  FLAT/unknown: buy RSI < 35, short RSI > 65
"""

from __future__ import annotations

from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, TrendRegime
from src.strategies.base import BaseStrategy


class RegimeAdaptiveStrategy(BaseStrategy):
    """EMA20 pullback in UP/DOWN regimes; RSI extremes in FLAT."""

    name: str = "regime_adaptive"

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 20))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))

    def on_bar(self, symbol: str, bar: Bar, symbol_state: SymbolState, market_state: MarketState) -> Signal | None:
        if not self._check_cooldown(symbol, bar.time):
            return None
        bars = list(symbol_state.bars_1m)
        if len(bars) < self.min_bars:
            return None

        closes = [float(b.close) for b in bars[-20:]]
        ema20 = sum(closes) / len(closes)  # simple MA as proxy

        # ATR from last 14 bars
        recent = bars[-14:]
        atr = sum(abs(float(b.high) - float(b.low)) for b in recent) / len(recent)
        if atr <= 0:
            return None

        price = float(bar.close)
        snapshot = market_state.regime_snapshot
        trend = snapshot.trend if snapshot else None

        if trend == TrendRegime.UP:
            if price > ema20 * 0.999:  # at or above EMA — no pullback
                return None
            side = OrderSide.BUY
        elif trend == TrendRegime.DOWN:
            if price < ema20 * 1.001:  # at or below EMA — no bounce
                return None
            side = OrderSide.SELL
        else:
            rsi_period = 14
            if len(bars) < rsi_period + 1:
                return None
            gains = [max(0.0, float(bars[-i].close) - float(bars[-i - 1].close)) for i in range(1, rsi_period + 1)]
            losses = [max(0.0, float(bars[-i - 1].close) - float(bars[-i].close)) for i in range(1, rsi_period + 1)]
            avg_gain = sum(gains) / rsi_period
            avg_loss = sum(losses) / rsi_period
            rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) if avg_loss > 0 else 50.0
            if rsi < 35.0:
                side = OrderSide.BUY
            elif rsi > 65.0:
                side = OrderSide.SELL
            else:
                return None

        stop = price - self.stop_atr_mult * atr if side == OrderSide.BUY else price + self.stop_atr_mult * atr
        target = price + self.target_atr_mult * atr if side == OrderSide.BUY else price - self.target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(symbol, side, bar, market_state, stop, target)
