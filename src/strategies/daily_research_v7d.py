"""Regime-Adaptive Keltner + BB Dual-Gate Reversion.

Long-only mean reversion with dual entry gates:
  Primary: Keltner Channel (EMA ± ATR*mult) per regime
  Secondary: Bollinger Band %B < 0 (below lower BB)
  Confirmation: IBS < 0.4 (close near low of bar)

Regime logic:
  UP   — Keltner OR BB entry, above SMA(50)
  FLAT — Keltner OR BB entry
  DOWN — require BOTH Keltner AND BB (tighter filter)

The dual-gate allows more trades in favorable regimes (UP/FLAT)
while maintaining selectivity in DOWN.

Skip: SHOCK vol, earnings, FOMC.
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
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.atr_period = int(config.get("atr_period", 14))
        # Keltner channel multipliers per regime
        self.kc_mult_up = float(config.get("kc_mult_up", 1.5))
        self.kc_mult_flat = float(config.get("kc_mult_flat", 2.0))
        self.kc_mult_down = float(config.get("kc_mult_down", 2.5))
        # Stop/target
        self.base_stop_atr_mult = float(config.get("base_stop_atr_mult", 1.5))
        self.base_target_atr_mult = float(config.get("base_target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        # Volume filter
        self.vol_lookback = int(config.get("vol_lookback", 20))
        self.vol_max_ratio = float(config.get("vol_max_ratio", 1.2))
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
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        subset = values[-period:]
        mean = sum(subset) / period
        variance = sum((v - mean) ** 2 for v in subset) / period
        return variance**0.5

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
    def _has_down_days(bars: list[Bar], n: int) -> bool:
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

        # Bollinger Band
        bb_sma = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_sma is None or bb_std is None or bb_std < 1e-9:
            return None
        lower_bb = bb_sma - self.bb_std * bb_std
        below_bb = bar.close < lower_bb

        # Read regime
        regime_labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = regime_labels.get("regime_trend", "FLAT").upper()
        regime_vol = regime_labels.get("regime_vol", "NORMAL").upper()

        # Skip SHOCK vol
        if regime_vol == "SHOCK":
            return None

        # Skip earnings/FOMC
        if regime_labels.get("near_earnings", False):
            return None
        if regime_labels.get("near_fomc", False):
            return None

        # Regime-dependent Keltner channel
        if regime_trend == "UP":
            kc_mult = self.kc_mult_up
        elif regime_trend == "DOWN":
            kc_mult = self.kc_mult_down
        else:
            kc_mult = self.kc_mult_flat

        lower_kc = ema - kc_mult * atr
        below_kc = bar.close <= lower_kc

        # Dual-gate entry logic per regime
        if regime_trend == "UP":
            # Either KC or BB, must be above SMA(50), require 1 down day
            if bar.close < sma50:
                return None
            if not self._has_down_days(bars, 1):
                return None
            if not (below_kc or below_bb):
                return None
        elif regime_trend == "DOWN":
            # Most selective: require BOTH KC and BB, 2 down days
            if not self._has_down_days(bars, 2):
                return None
            if not (below_kc and below_bb):
                return None
        else:
            # FLAT: either KC or BB, 1 down day
            if not self._has_down_days(bars, 1):
                return None
            if not (below_kc or below_bb):
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
                "below_kc": below_kc,
                "below_bb": below_bb,
                "ema": round(ema, 2),
                "sma50": round(sma50, 2),
                "seed": "regime_switch",
            },
        )
