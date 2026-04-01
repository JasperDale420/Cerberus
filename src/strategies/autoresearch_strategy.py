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

        atr = mtf.get_atr("1m", 14)
        if atr is None or atr <= 0:
            return None

        # Pullback zone: price at or slightly below EMA20 (within 1 ATR)
        dist_from_ema = ema20 - bar.close
        if dist_from_ema < 0:
            return None  # price above EMA20, no pullback
        if dist_from_ema > 1.0 * atr:
            return None  # too deep — possible breakdown

        # RSI filter: 30-55 (genuine pullback, not overbought)
        rsi = mtf.get_rsi("1m", 14)
        if rsi is None or rsi > 55 or rsi < 30:
            return None

        # Volume confirmation: bar volume above 20-bar average
        bars = symbol_state.bars_1m
        if len(bars) >= 20:
            vol_sum = sum(float(b.volume) for b in list(bars)[-20:])
            vol_avg = vol_sum / 20.0
            if vol_avg > 0 and bar.volume < 1.2 * vol_avg:
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
            meta={"reason": "ema_pullback_vol", "rsi": round(rsi, 1)},
        )
