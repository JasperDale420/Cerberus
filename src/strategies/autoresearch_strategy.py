"""Autoresearch Strategy — the file you modify."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy


class AutoresearchStrategy(BaseStrategy):
    name = "autoresearch_strategy"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.min_bars = int(config.get("min_bars", 30))

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

        mtf = MultiTimeframeAnalyzer(symbol_state)

        ema20 = mtf.get_ema("1m", 20)
        ema50 = mtf.get_ema("1m", 50)
        if ema20 is None or ema50 is None:
            return None

        # Uptrend: EMA20 > EMA50
        if ema20 <= ema50:
            return None

        # Pullback: price at or below EMA20
        if bar.close > ema20:
            return None

        rsi = mtf.get_rsi("1m", 14)
        if rsi is not None and (rsi > 65 or rsi < 25):
            return None

        atr = mtf.get_atr("1m", 14)
        if atr is None or atr <= 0:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = bar.close + self.target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop,
            target,
            meta={"reason": "ema_pullback"},
        )
