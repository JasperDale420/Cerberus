"""Volatility Expansion + Trend Confirmation.

Evolved from seed_vol_breakout. Buy when:
1. ATR is expanding (current > average) - volatility expansion
2. Price is trending (close > EMA) - trend confirmation
3. Close above recent high channel - breakout confirmed
4. Volume above average - participation
5. Regime filter: skip SHOCK vol, skip earnings

Long-only, daily bars. Adaptive stop/target based on ATR.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v9c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 30))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_avg_period = int(config.get("atr_avg_period", 20))
        self.atr_expansion_mult = float(config.get("atr_expansion_mult", 1.2))
        self.breakout_lookback = int(config.get("breakout_lookback", 5))
        self.ema_period = int(config.get("ema_period", 10))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 1.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 7))

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

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = values[-period]
        for v in values[-period + 1 :]:
            ema = v * mult + ema * (1 - mult)
        return ema

    def _atr_series(self, bars: list[Bar], period: int, count: int) -> list[float]:
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

        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings"):
            return None

        # Skip SHOCK vol regime
        regime_vol = regime_labels.get("regime_vol", "NORMAL")
        if regime_vol == "SHOCK":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # Current ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # ATR expansion: current ATR > mult * its average
        atr_values = self._atr_series(bars, self.atr_period, self.atr_avg_period)
        if len(atr_values) < self.atr_avg_period:
            return None
        atr_avg = sum(atr_values) / len(atr_values)
        if atr_avg < 1e-9 or atr < self.atr_expansion_mult * atr_avg:
            return None

        # EMA trend filter: close must be above EMA
        ema = self._ema(closes, self.ema_period)
        if ema is None or bar.close <= ema:
            return None

        # Breakout: close above N-day high (excluding current bar)
        if len(bars) < self.breakout_lookback + 1:
            return None
        lookback_highs = [b.high for b in bars[-(self.breakout_lookback + 1) : -1]]
        high_nd = max(lookback_highs)
        if bar.close <= high_nd:
            return None

        # Volume confirmation
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # Stop and target
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
                "atr_avg": round(atr_avg, 4),
                "atr_expansion": round(atr / atr_avg, 2),
                "high_nd": round(high_nd, 2),
                "ema": round(ema, 2),
                "vol_ratio": round(bar.volume / avg_vol, 2),
                "regime_vol": regime_vol,
                "seed": "vol_breakout_v2",
            },
        )
