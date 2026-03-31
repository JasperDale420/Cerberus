"""
RegimeTrendUp Strategy

Type: trend-following
Description: Buys pullbacks in uptrends during UP+NORMAL regime.
3-factor entry: EMA trend confirmation, pullback to EMA20, RSI not overbought.
BUY-only.
"""

from __future__ import annotations

from typing import Any

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy


class RegimeTrendUpStrategy(BaseStrategy):
    """Buy pullbacks in uptrends: EMA trend + price near EMA20 + RSI range."""

    name: str = "regime_trend_up"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 30))
        self.pullback_pct = float(config.get("pullback_pct", 0.008))
        self.rsi_max = float(config.get("rsi_max", 70.0))
        self.rsi_min = float(config.get("rsi_min", 35.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_rr = float(config.get("target_rr", 2.5))
        self.time_window_start = time_utils.parse_time_string(str(config.get("time_window_start", "09:35")))
        self.time_window_end = time_utils.parse_time_string(str(config.get("time_window_end", "15:45")))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        if not self._is_evaluation_bar(bar):
            return None

        t = time_utils.get_eastern_time_of_day(bar.time)
        if not (self.time_window_start <= t <= self.time_window_end):
            return None

        if not self._check_cooldown(symbol, bar.time):
            return None

        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        if symbol_state.position is not None:
            return None

        mtf = MultiTimeframeAnalyzer(symbol_state)

        # Factor 1: EMA20 > EMA50 — intraday uptrend confirmed
        ema20 = mtf.get_ema("1m", 20)
        ema50 = mtf.get_ema("1m", 50)
        if ema20 is None or ema50 is None or ema20 <= ema50:
            return None

        # Factor 2: Price pulling back to EMA20 zone (not chasing)
        dist_pct = (bar.close - ema20) / ema20
        if dist_pct < -self.pullback_pct or dist_pct > self.pullback_pct:
            return None

        # Factor 3: RSI not overbought and not in freefall
        rsi = mtf.get_rsi("1m", 14)
        if rsi is not None and (rsi > self.rsi_max or rsi < self.rsi_min):
            return None

        atr = mtf.get_atr("1m", 14)
        if atr is None or atr <= 0:
            return None

        stop_price = bar.close - self.stop_atr_mult * atr
        target_price = bar.close + self.target_rr * (bar.close - stop_price)

        return self._create_signal(
            symbol=symbol,
            side=OrderSide.BUY,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=1.0,
            meta={"ema20": round(ema20, 4), "dist_pct": round(dist_pct, 5), "rsi": round(rsi, 2) if rsi else None},
        )
