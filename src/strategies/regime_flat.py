"""
RegimeFlat Strategy

Flat-regime specialist: bidirectional RSI mean-reversion.
Trades only when the live MarketContextService (Labeler B) reports
TrendRegime.FLAT — i.e. no directional bias, expect range-bound action.
BUY oversold, SELL overbought. ATR-based stops/targets.

Note: gates exclusively on market_state.regime_snapshot.trend
(populated by Labeler B in live mode); does NOT read
symbol_state.meta["regime_labels"] (Labeler A backtest-only dict).
"""

from __future__ import annotations

from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, TrendRegime
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy


class RegimeFlatStrategy(BaseStrategy):
    """Bidirectional RSI mean-reversion gated on TrendRegime.FLAT."""

    name: str = "regime_flat"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.rsi_oversold = float(config.get("rsi_oversold", 30.0))
        self.rsi_overbought = float(config.get("rsi_overbought", 70.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.min_bars = int(config.get("min_bars", 20))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None
        if self.is_past_hard_stop(bar.time):
            return None

        snapshot = market_state.regime_snapshot
        if snapshot is None or snapshot.trend != TrendRegime.FLAT:
            return None

        if symbol_state.position is not None:
            return None

        mtf = MultiTimeframeAnalyzer(symbol_state)
        rsi = mtf.get_rsi("1m", 14)
        if rsi is None:
            return None

        if rsi < self.rsi_oversold:
            side = OrderSide.BUY
        elif rsi > self.rsi_overbought:
            side = OrderSide.SELL
        else:
            return None

        atr = mtf.get_atr("1m", 14)
        if atr is None or atr <= 0:
            return None

        stop_distance = self.stop_atr_mult * atr
        target_distance = self.target_atr_mult * atr

        if side == OrderSide.BUY:
            stop_price = bar.close - stop_distance
            target_price = bar.close + target_distance
        else:
            stop_price = bar.close + stop_distance
            target_price = bar.close - target_distance

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={"rsi_1m": round(rsi, 2), "trend": "FLAT"},
        )
