"""Regime-Adaptive Keltner + IBS Dip Buy (long-only).

Entry: price below lower Keltner + low IBS + consecutive down closes.
Regime-adaptive parameters:
  UP   — tighter KC entry (1.5 ATR), stricter IBS (0.20), bigger target (EMA)
  DOWN — standard KC (2.0 ATR), IBS 0.30, tight target (1.5 ATR)
  FLAT — standard KC (2.0 ATR), IBS 0.30, EMA target

Skips SHOCK vol, earnings, FOMC. Skip DOWN+HIGH.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedRegimeSwitchStrategy(BaseStrategy):
    name = "daily_research_v8d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.kc_ema_period = int(config.get("kc_ema_period", 20))
        self.atr_period = int(config.get("atr_period", 14))
        # Regime-adaptive KC multipliers
        self.kc_mult_up = float(config.get("kc_mult_up", 1.5))
        self.kc_mult_default = float(config.get("kc_mult_default", 2.0))
        # Regime-adaptive IBS thresholds
        self.ibs_max_up = float(config.get("ibs_max_up", 0.20))
        self.ibs_max_default = float(config.get("ibs_max_default", 0.30))
        self.consec_down_min = int(config.get("consec_down_min", 2))
        # Stop/target
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.down_target_atr_mult = float(config.get("down_target_atr_mult", 1.5))
        self.default_target_atr_mult = float(config.get("default_target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicator helpers ---

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
    def _consecutive_downs(closes: list[float]) -> int:
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _ibs(bar: Bar) -> float:
        rng = bar.high - bar.low
        if rng < 1e-9:
            return 0.5
        return (bar.close - bar.low) / rng

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

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        # Regime
        regime_trend = labels.get("regime_trend", "FLAT").upper()
        regime_vol = labels.get("regime_vol", "NORMAL").upper()

        # Skip DOWN+HIGH
        if regime_trend == "DOWN" and regime_vol == "HIGH":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Core indicators
        ema = self._ema(closes, self.kc_ema_period)
        atr = self._atr(bars, self.atr_period)
        if ema is None or atr is None or atr < 1e-9:
            return None

        # Regime-adaptive parameters
        if regime_trend == "UP":
            kc_mult = self.kc_mult_up
            ibs_max = self.ibs_max_up
        else:
            kc_mult = self.kc_mult_default
            ibs_max = self.ibs_max_default

        # Keltner Channel lower band
        lower_kc = ema - kc_mult * atr

        # Entry condition 1: price at or below lower KC
        if bar.close > lower_kc:
            return None

        # Entry condition 2: low IBS
        ibs = self._ibs(bar)
        if ibs > ibs_max:
            return None

        # Entry condition 3: consecutive down closes
        consec = self._consecutive_downs(closes)
        if consec < self.consec_down_min:
            return None

        # Stop and target (regime-adaptive)
        stop = bar.close - self.stop_atr_mult * atr

        if regime_trend == "UP" and ema > bar.close:
            # UP: target EMA (the trend should pull price back to the mean)
            target = ema
        elif regime_trend == "FLAT" and ema > bar.close:
            # FLAT: target EMA (mean reversion)
            target = ema
        elif regime_trend == "DOWN":
            # DOWN: tight target (quick bounce, don't overstay)
            target = bar.close + self.down_target_atr_mult * atr
        else:
            target = bar.close + self.default_target_atr_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "regime": f"{regime_trend}+{regime_vol}",
                "kc_dist": round((lower_kc - bar.close) / atr, 2),
                "ibs": round(ibs, 3),
                "consec": consec,
                "seed": "keltner_regime_adaptive",
            },
        )
