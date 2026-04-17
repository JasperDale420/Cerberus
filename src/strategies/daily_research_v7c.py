"""Keltner Channel Breakout with Regime-Adaptive Stops (vol_breakout archetype).

Buy when price breaks above upper Keltner Channel with volume confirmation.
Regime-adaptive: tighter stops in DOWN/HIGH, wider in UP/NORMAL.
Skip Fridays (historically negative). Short hold period for consistency.
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
        self.min_bars = int(config.get("min_bars", 30))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.keltner_period = int(config.get("keltner_period", 20))
        self.keltner_mult = float(config.get("keltner_mult", 1.5))
        self.atr_expansion = float(config.get("atr_expansion", 1.2))

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
        ema = values[0]
        for v in values[1:]:
            ema = v * mult + ema * (1 - mult)
        return ema

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

        # Skip Fridays — historically negative PnL
        if hasattr(bar.time, "weekday") and bar.time.weekday() == 4:
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        atr = self._atr(bars, 14)
        if atr is None or atr < 1e-9:
            return None

        if len(closes) < self.keltner_period + 5:
            return None

        # Keltner Channel: EMA +/- mult * ATR
        ema_mid = self._ema(closes[-self.keltner_period :], self.keltner_period)
        if ema_mid is None:
            return None

        upper_keltner = ema_mid + self.keltner_mult * atr
        ema_mid - self.keltner_mult * atr

        # ATR expansion: current ATR vs prior ATR (volatility increasing)
        atr_prev = self._atr(bars[:-5], 14)
        if atr_prev is None or atr_prev < 1e-9:
            return None
        atr_ratio = atr / atr_prev

        # Entry: price breaks above upper Keltner + ATR expanding
        if bar.close <= upper_keltner:
            return None
        if atr_ratio < self.atr_expansion:
            return None

        # Volume confirmation: above 20-day average
        volumes = [b.volume for b in bars]
        if len(volumes) < 20:
            return None
        avg_vol = sum(volumes[-20:]) / 20
        if avg_vol <= 0 or bar.volume < avg_vol:
            return None

        # Close in upper third of range (strong close)
        today_range = bar.high - bar.low
        if today_range < 1e-9:
            return None
        close_pos = (bar.close - bar.low) / today_range
        if close_pos < 0.6:
            return None

        # Regime-adaptive stops
        regime_trend = regime_labels.get("regime_trend", "UP")
        regime_vol = regime_labels.get("regime_vol", "NORMAL")
        if regime_trend == "DOWN" or regime_vol == "HIGH":
            stop_mult = self.stop_atr_mult * 0.7
            target_mult = self.target_atr_mult * 0.8
        else:
            stop_mult = self.stop_atr_mult
            target_mult = self.target_atr_mult

        stop = bar.close - stop_mult * atr
        target = bar.close + target_mult * atr

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
                "atr_ratio": round(atr_ratio, 2),
                "close_pos": round(close_pos, 2),
                "keltner_break": round((bar.close - upper_keltner) / atr, 2),
                "seed": "vol_breakout",
            },
        )
