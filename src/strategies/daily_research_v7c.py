"""Wide-Range Bar Continuation (evolved from vol_breakout seed).

Buy after wide-range up bars (volatility expansion with bullish close).
Zero tunable params — structural edge must work across all regimes.
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
        self.min_bars = int(config.get("min_bars", 25))
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

        # ATR(14) for volatility reference
        atr = self._atr(bars, 14)
        if atr is None or atr < 1e-9:
            return None

        # Wide range bar: today's range > 1.3x ATR
        today_range = bar.high - bar.low
        if today_range < 1e-9 or today_range < 1.3 * atr:
            return None

        # Close strength: close in top 30% of today's range
        close_position = (bar.close - bar.low) / today_range
        if close_position < 0.7:
            return None

        # Up day: close above previous close
        prev_close = bars[-2].close
        if bar.close <= prev_close:
            return None

        # Trend: close above SMA(20)
        sma20 = self._sma(closes, 20)
        if sma20 is None or bar.close <= sma20:
            return None

        # Anti-exhaustion: don't buy if close > 3 ATR above SMA(20)
        if bar.close > sma20 + 3.0 * atr:
            return None

        stop = bar.close - self.stop_atr_mult * atr
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
                "range_mult": round(today_range / atr, 2),
                "close_pos": round(close_position, 2),
                "seed": "vol_breakout",
            },
        )
