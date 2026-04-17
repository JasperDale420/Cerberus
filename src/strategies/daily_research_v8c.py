"""Seed: Volatility Breakout.

Buy when ATR expands, price breaks 10-day high, and volume confirms.
Skips near-earnings symbols. Long-only, daily bars, max_hold_days=7.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v8c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_avg_period = int(config.get("atr_avg_period", 20))
        self.atr_expansion_mult = float(config.get("atr_expansion_mult", 1.5))
        self.breakout_lookback = int(config.get("breakout_lookback", 10))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 7))

    # --- Indicator helpers ---

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

    def _atr_series(self, bars: list[Bar], period: int, count: int) -> list[float]:
        """Compute a series of ATR values for the last `count` bars."""
        result = []
        for i in range(count):
            end_idx = len(bars) - count + i + 1
            if end_idx < period + 1:
                continue
            sub = bars[:end_idx]
            val = self._atr(sub, period)
            if val is not None:
                result.append(val)
        return result

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

        # Skip near-earnings symbols
        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings"):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # Current ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # ATR expansion: current ATR > 1.5x its 20-day average
        atr_values = self._atr_series(bars, self.atr_period, self.atr_avg_period)
        if len(atr_values) < self.atr_avg_period:
            return None
        atr_avg = sum(atr_values) / len(atr_values)
        if atr_avg < 1e-9 or atr < self.atr_expansion_mult * atr_avg:
            return None

        # Breakout: close above 10-day high (excluding current bar)
        if len(bars) < self.breakout_lookback + 1:
            return None
        lookback_highs = [b.high for b in bars[-(self.breakout_lookback + 1) : -1]]
        high_10d = max(lookback_highs)
        if bar.close <= high_10d:
            return None

        # Volume confirmation: > 1.5x 20-day average
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # Stop below breakout bar's low, target 2x ATR above entry
        stop = bar.low
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
                "atr_avg": round(atr_avg, 4),
                "atr_expansion": round(atr / atr_avg, 2),
                "high_10d": round(high_10d, 2),
                "vol_ratio": round(bar.volume / avg_vol, 2),
                "seed": "vol_breakout",
            },
        )
