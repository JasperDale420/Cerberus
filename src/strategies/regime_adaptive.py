"""
RegimeAdaptive Strategy

Type: adaptive
Description: Cross-regime generalist — switches entry mode based on trend regime.
  UP: buys RSI dips (momentum continuation)
  DOWN: shorts RSI bounces (fade the bounce)
  FLAT/unknown: RSI extremes (mean reversion)
"""

from __future__ import annotations

from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, TrendRegime
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy


class RegimeAdaptiveStrategy(BaseStrategy):
    """Switch entry logic by trend regime: dip-buy in UP, fade bounce in DOWN, RSI extremes in FLAT."""

    name: str = "regime_adaptive"

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.rsi_long_entry = float(config.get("rsi_long_entry", 45.0))
        self.rsi_short_entry = float(config.get("rsi_short_entry", 60.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.min_bars = int(config.get("min_bars", 20))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        if not self._is_evaluation_bar(bar):
            return None
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        mtf = MultiTimeframeAnalyzer(symbol_state)
        rsi = mtf.get_rsi("1m", 14)
        atr = mtf.get_atr("1m", 14)
        if rsi is None or atr is None or atr <= 0:
            return None

        snapshot = market_state.regime_snapshot
        trend = snapshot.trend if snapshot else None

        if trend == TrendRegime.UP:
            if rsi >= self.rsi_long_entry:
                return None
            side = OrderSide.BUY
        elif trend == TrendRegime.DOWN:
            if rsi <= self.rsi_short_entry:
                return None
            side = OrderSide.SELL
        else:  # FLAT or no snapshot — mean reversion
            if rsi < 35.0:
                side = OrderSide.BUY
            elif rsi > 65.0:
                side = OrderSide.SELL
            else:
                return None

        stop_dist = self.stop_atr_mult * atr
        if side == OrderSide.BUY:
            stop_price = bar.close - stop_dist
            target_price = bar.close + self.target_atr_mult * atr
        else:
            stop_price = bar.close + stop_dist
            target_price = bar.close - self.target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(symbol, side, bar, market_state, stop_price, target_price)
