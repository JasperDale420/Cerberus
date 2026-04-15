"""IBS + Donchian Hybrid with Max-Loss Clamp.

Two modes:
  1. IBS Mean Reversion — buy when IBS < 0.2 (close near day low), price above SMA.
  2. Donchian Breakout — new N-day high with ATR expansion.

Key: max_loss_pct clamps stop loss to prevent catastrophic single-trade losses.
This caps downside per trade, making worst-window PF more consistent.

Long-only, daily bars.
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
        self.min_bars = int(config.get("min_bars", 50))
        self.atr_period = int(config.get("atr_period", 14))

        # Optimizable
        self.atr_expansion = float(config.get("atr_expansion", 1.3))
        self.vol_surge_mult = float(config.get("vol_surge_mult", 1.3))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))

        # IBS params
        self.ibs_threshold = float(config.get("ibs_threshold", 0.2))
        self.sma_period = int(config.get("sma_period", 20))

        # Structural
        self.breakout_lookback = int(config.get("breakout_lookback", 10))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.max_loss_pct = float(config.get("max_loss_pct", 0.03))

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

    def _clamp_stop(self, price: float, stop: float) -> float:
        """Never lose more than max_loss_pct per trade."""
        min_stop = price * (1.0 - self.max_loss_pct)
        return max(stop, min_stop)

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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        vol_regime = regime_labels.get("regime_vol", "NORMAL")
        if vol_regime == "SHOCK":
            return None

        trend = regime_labels.get("regime_trend", "FLAT")
        if trend == "DOWN":
            return None

        # Mode 1: IBS Mean Reversion
        signal = self._try_ibs(symbol, bar, closes, atr, market_state)
        if signal is not None:
            return signal

        # Mode 2: Donchian Breakout
        signal = self._try_donchian(symbol, bar, bars, closes, volumes, atr, market_state)
        return signal

    def _try_ibs(
        self,
        symbol: str,
        bar: Bar,
        closes: list[float],
        atr: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None

        ibs = (bar.close - bar.low) / bar_range
        if ibs > self.ibs_threshold:
            return None

        sma = self._sma(closes, self.sma_period)
        if sma is None:
            return None
        if bar.close < sma - atr:
            return None

        if bar.close < sma:
            target = sma
        else:
            target = bar.close + self.target_atr_mult * atr

        stop = self._clamp_stop(bar.close, bar.close - self.stop_atr_mult * atr)

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={"mode": "IBS", "ibs": round(ibs, 3)},
        )

    def _try_donchian(
        self,
        symbol: str,
        bar: Bar,
        bars: list[Bar],
        closes: list[float],
        volumes: list[float],
        atr: float,
        market_state: MarketState,
    ) -> Optional[Signal]:
        if len(bars) < self.breakout_lookback + 1:
            return None

        lookback_highs = [b.high for b in bars[-(self.breakout_lookback + 1) : -1]]
        if bar.close <= max(lookback_highs):
            return None

        # ATR expansion
        atr_values = self._atr_series(bars, self.atr_period, 20)
        if len(atr_values) >= 10:
            atr_avg = sum(atr_values) / len(atr_values)
            if atr_avg > 1e-9 and atr / atr_avg < self.atr_expansion:
                return None

        # Volume confirmation
        avg_vol = self._sma(volumes, 20)
        if avg_vol is not None and avg_vol > 1e-9:
            if bar.volume / avg_vol < self.vol_surge_mult:
                return None

        stop = self._clamp_stop(bar.close, bar.close - self.stop_atr_mult * atr)
        target = bar.close + self.target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={"mode": "DONCHIAN"},
        )
