"""UP+NORMAL regime specialist — buys EMA pullbacks in confirmed uptrends.

Type: trend-following
Description: BUY-only strategy for UP+NORMAL (trending bull, low vol) regimes.
Entry: EMA9 > EMA20 on 1m + RSI momentum zone (30-70) + price above VWAP
       + bullish candle confirmation (close in upper half of bar range).
Exit: ATR-based stop/target with trailing stop.
"""

from __future__ import annotations

from typing import Any

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy


class RegimeTrendUpStrategy(BaseStrategy):
    """UP+NORMAL specialist: buy momentum continuations in uptrends.

    Four entry conditions:
    1. EMA9 > EMA20 on 1m (confirmed uptrend)
    2. RSI in momentum zone 30-70 (not extreme either way — avoids overbought chasing)
    3. Price above VWAP (long-side bias, skipped if VWAP unavailable)
    4. Bullish candle: close in upper half of bar range (buying pressure confirmation)
    """

    name: str = "regime_trend_up"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 20))
        self.time_window_start = time_utils.parse_time_string(str(config.get("time_window_start", "09:35")))
        self.time_window_end = time_utils.parse_time_string(str(config.get("time_window_end", "15:45")))
        overrides = config.get("_optuna_overrides", {})
        self.rsi_entry_low = float(overrides.get("rsi_entry_low", config.get("rsi_entry_low", 30.0)))
        self.rsi_entry_high = float(overrides.get("rsi_entry_high", config.get("rsi_entry_high", 70.0)))
        self.stop_atr_mult = float(overrides.get("stop_atr_mult", config.get("stop_atr_mult", 1.5)))
        self.target_atr_mult = float(overrides.get("target_atr_mult", config.get("target_atr_mult", 3.0)))
        self.max_hold_minutes = int(overrides.get("max_hold_minutes", config.get("max_hold_minutes", 90)))
        # Disable base higher-TF gate — we use our own EMA check inline
        self.higher_tf_alignment = False

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

        t = time_utils.get_eastern_time_of_day(bar.time)
        if not (self.time_window_start <= t <= self.time_window_end):
            return None

        mtf = MultiTimeframeAnalyzer(symbol_state)

        # Gate 1: uptrend confirmed — EMA9 > EMA20 on 1m
        ema9 = mtf.get_ema("1m", 9)
        ema20 = mtf.get_ema("1m", 20)
        if ema9 is None or ema20 is None:
            return None
        if ema9 <= ema20:
            return None

        # Gate 2: RSI pullback zone — dip in uptrend, not overbought
        rsi = mtf.get_rsi("1m", 14)
        if rsi is None:
            return None
        if not (self.rsi_entry_low <= rsi <= self.rsi_entry_high):
            return None

        # Gate 3: price above VWAP — long-side bias
        if bar.vwap > 0 and bar.close < bar.vwap:
            return None

        # Gate 4: bullish candle — close in upper half of range (buying pressure)
        bar_range = bar.high - bar.low
        if bar_range > 0 and bar.close < (bar.low + bar_range * 0.5):
            return None

        # ATR for stop/target sizing
        atr = mtf.get_atr("1m", 14)
        if atr is None or atr <= 0:
            return None

        stop_price = bar.close - self.stop_atr_mult * atr
        target_price = bar.close + self.target_atr_mult * atr

        meta = {
            "strategy_version": "regime_trend_up_v1",
            "ema9": round(ema9, 4),
            "ema20": round(ema20, 4),
            "rsi_1m": round(rsi, 2),
            "atr_1m": round(atr, 6),
            "exit_config": {
                "max_hold_minutes": self.max_hold_minutes,
                "trailing_enabled": True,
                "trail_timeframe": "1m",
                "trail_lookback": 3,
                "trail_min_profit_r": 0.5,
            },
        }

        return self._create_signal(
            symbol=symbol,
            side=OrderSide.BUY,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta=meta,
        )
