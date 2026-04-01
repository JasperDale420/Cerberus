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

        # Regime filter: skip low-vol uptrends (consistent losers)
        meta = symbol_state.meta
        trend = meta.get("regime_trend", "")
        vol = meta.get("regime_vol", "")
        if trend == "UP" and vol in ("LOW", "NORMAL"):
            return None

        mtf = MultiTimeframeAnalyzer(symbol_state)

        # 1m uptrend: EMA20 > EMA50
        ema20 = mtf.get_ema("1m", 20)
        ema50 = mtf.get_ema("1m", 50)
        if ema20 is None or ema50 is None or ema20 <= ema50:
            return None

        atr = mtf.get_atr("1m", 14)
        if atr is None or atr <= 0:
            return None

        # ADX trend strength: require confirmed trend
        adx = mtf.get_adx("1m")
        if adx is not None and adx < 20:
            return None

        # Tight pullback: price within 0.5 ATR below EMA20
        dist = ema20 - bar.close
        if dist < 0 or dist > 0.5 * atr:
            return None

        # RSI: 30-55
        rsi = mtf.get_rsi("1m", 14)
        if rsi is None or rsi > 55 or rsi < 30:
            return None

        # Regime-adaptive stops
        stop_dist = self._apply_regime_volatility_multiplier(self.stop_atr_mult * atr, market_state)
        target_dist = self._apply_regime_volatility_multiplier(self.target_atr_mult * atr, market_state)
        stop = bar.close - stop_dist
        target = bar.close + target_dist

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop,
            target,
            meta={"reason": "ema_pullback_tight", "rsi": round(rsi, 1)},
        )
