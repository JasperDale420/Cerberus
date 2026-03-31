"""Autoresearch Strategy — the file you modify.

This is the ONLY file the autoresearch agent edits. Everything else is frozen.
Start simple. Iterate. The evaluation is fixed at ~30 min per run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
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

        # Read indicators
        ema20 = symbol_state.indicators.get("ema_close_1m:20")
        ema50 = symbol_state.indicators.get("ema_close_1m:50")
        rsi = symbol_state.indicators.get("rsi_1m:14")
        atr = symbol_state.indicators.get("atr_1m:14")
        if ema20 is None or ema50 is None or rsi is None or atr is None:
            return None

        ema20 = float(ema20)
        ema50 = float(ema50)
        rsi = float(rsi)
        atr = float(atr)
        if atr < 0.01:
            return None

        # Uptrend: EMA20 > EMA50
        if ema20 <= ema50:
            return None

        # Pullback: price near or below EMA20
        if bar.close > ema20:
            return None

        # RSI filter: not overbought, not too oversold (falling knife)
        if rsi > 60 or rsi < 25:
            return None

        # Entry: BUY
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
            meta={"reason": "ema_pullback", "rsi": rsi, "atr": atr},
        )
