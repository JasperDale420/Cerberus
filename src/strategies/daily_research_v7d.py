"""Regime-Adaptive Keltner Channel Reversion.

Long-only mean reversion using Keltner Channel (EMA ± ATR*mult):
  UP   — buy at lower Keltner band with volume exhaustion, above SMA(50)
  FLAT — buy at lower Keltner, target mid-channel (EMA)
  DOWN — buy only extreme lower Keltner (wider mult), require 2+ down days

No shorts. Regime skips: DOWN+HIGH/SHOCK.
Stops/targets ATR-based. Daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedRegimeSwitchStrategy(BaseStrategy):
    name = "daily_research_v7d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.ema_period = int(config.get("ema_period", 20))
        self.sma_period = int(config.get("sma_period", 50))
        self.atr_period = int(config.get("atr_period", 14))
        # Keltner channel multipliers per regime
        self.kc_mult_up = float(config.get("kc_mult_up", 1.5))
        self.kc_mult_flat = float(config.get("kc_mult_flat", 2.0))
        self.kc_mult_down = float(config.get("kc_mult_down", 2.5))
        # Stop/target
        self.base_stop_atr_mult = float(config.get("base_stop_atr_mult", 1.5))
        self.base_target_atr_mult = float(config.get("base_target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        # Volume filter
        self.vol_lookback = int(config.get("vol_lookback", 20))
        self.vol_max_ratio = float(config.get("vol_max_ratio", 1.2))
        # Consecutive down days for DOWN regime
        self.min_down_days = int(config.get("min_down_days", 2))
        # Max ATR as pct of price (skip very volatile names)
        self.max_atr_pct = float(config.get("max_atr_pct", 0.04))
        # Max stop as pct of price
        self.max_stop_pct = float(config.get("max_stop_pct", 0.03))

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
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = (v - ema) * mult + ema
        return ema

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
    def _consecutive_down_days(bars: list[Bar], n: int) -> bool:
        if len(bars) < n + 1:
            return False
        for i in range(-n, 0):
            if bars[i].close >= bars[i - 1].close:
                return False
        return True

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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Core indicators
        ema = self._ema(closes, self.ema_period)
        sma50 = self._sma(closes, self.sma_period)
        atr = self._atr(bars, self.atr_period)

        if any(v is None for v in (ema, sma50, atr)) or atr < 1e-9:
            return None

        # Skip extremely volatile names
        if atr / bar.close > self.max_atr_pct:
            return None

        # Read regime
        regime_labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = regime_labels.get("regime_trend", "FLAT").upper()
        regime_vol = regime_labels.get("regime_vol", "NORMAL").upper()

        # Skip DOWN+HIGH/SHOCK
        if regime_trend == "DOWN" and regime_vol in ("HIGH", "SHOCK"):
            return None

        # Skip earnings/FOMC
        if regime_labels.get("near_earnings", False):
            return None
        if regime_labels.get("near_fomc", False):
            return None

        # Regime-dependent Keltner channel
        if regime_trend == "UP":
            kc_mult = self.kc_mult_up
            # Must be above SMA(50) in uptrend
            if bar.close < sma50:
                return None
        elif regime_trend == "DOWN":
            kc_mult = self.kc_mult_down
            # Require consecutive down days in DOWN regime
            if not self._consecutive_down_days(bars, self.min_down_days):
                return None
        else:
            kc_mult = self.kc_mult_flat

        lower_kc = ema - kc_mult * atr

        # Entry: price at or below lower Keltner band
        if bar.close > lower_kc:
            return None

        # Volume filter: skip if volume is spiking (panic selling, not exhaustion)
        if len(bars) >= self.vol_lookback:
            avg_vol = sum(b.volume for b in bars[-self.vol_lookback :]) / self.vol_lookback
            if avg_vol > 0 and bar.volume > avg_vol * self.vol_max_ratio:
                return None

        # IBS check — close should be in lower half of bar (not already recovering)
        bar_range = bar.high - bar.low
        if bar_range > 1e-9:
            ibs = (bar.close - bar.low) / bar_range
            if ibs > 0.4:
                return None

        # Stop and target
        raw_stop_dist = self.base_stop_atr_mult * atr
        max_stop_dist = bar.close * self.max_stop_pct
        stop_dist = min(raw_stop_dist, max_stop_dist)
        stop = bar.close - stop_dist
        target = bar.close + self.base_target_atr_mult * atr

        if target <= bar.close or stop >= bar.close:
            return None

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "regime": regime_trend,
                "regime_vol": regime_vol,
                "kc_mult": kc_mult,
                "lower_kc": round(lower_kc, 2),
                "ema": round(ema, 2),
                "sma50": round(sma50, 2),
                "seed": "regime_switch",
            },
        )
