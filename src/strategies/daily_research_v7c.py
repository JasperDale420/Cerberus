"""Keltner Channel Breakout Strategy.

Keltner Channel = EMA(20) ± ATR(14) * multiplier.
Upper channel break = volatility breakout + trend confirmation in ONE signal.
This naturally combines: price > trend (EMA), ATR expansion, and breakout.

Entry: close > upper Keltner AND close > yesterday's close (momentum confirm)
Stop: EMA or lower Keltner (whichever is closer)
Target: 2-3x ATR above entry

Long-only, daily bars. Skips DOWN regime and SHOCK vol.
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
        self.ema_period = int(config.get("ema_period", 20))

        # Optimizable
        self.atr_expansion = float(config.get("atr_expansion", 1.5))
        self.vol_surge_mult = float(config.get("vol_surge_mult", 1.2))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))

        # Keltner channel width (ATR multiplier for upper/lower bands)
        self.keltner_mult = float(config.get("keltner_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        self.breakout_lookback = int(config.get("breakout_lookback", 10))

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
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = v * mult + ema * (1 - mult)
        return ema

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

        vol_regime = regime_labels.get("regime_vol", "NORMAL")
        if vol_regime == "SHOCK":
            return None

        trend = regime_labels.get("regime_trend", "FLAT")
        if trend == "DOWN":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        ema = self._ema(closes, self.ema_period)
        if ema is None:
            return None

        # Keltner Channels
        upper_keltner = ema + self.keltner_mult * atr
        price = bar.close

        # Entry: close above upper Keltner channel
        if price <= upper_keltner:
            return None

        # Momentum confirmation: close > yesterday's close
        if len(closes) >= 2 and price <= closes[-2]:
            return None

        # Volume confirmation (optional — use softer filter)
        volumes = [b.volume for b in bars]
        avg_vol = self._sma(volumes, 20)
        if avg_vol is not None and avg_vol > 1e-9:
            vol_ratio = bar.volume / avg_vol
            if vol_ratio < self.vol_surge_mult:
                return None

        # Stop: EMA or ATR-based (whichever is higher = tighter)
        stop_ema = ema
        stop_atr = price - self.stop_atr_mult * atr
        stop = max(stop_ema, stop_atr)

        # Target
        target = price + self.target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "KELTNER",
                "keltner_dist": round((price - upper_keltner) / atr, 2),
                "trend": trend,
            },
        )
