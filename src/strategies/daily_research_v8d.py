"""Regime-Filtered Keltner + IBS Dip Buy (long-only).

Unified entry logic across all regimes:
  1. Price at or below lower Keltner Channel (EMA - mult*ATR)
  2. Low IBS (closed near day's low — seller exhaustion)
  3. At least N consecutive down closes (momentum exhaustion)

Regime used as FILTER only — skip bad regime combos (DOWN+HIGH).
Skips SHOCK vol, earnings, FOMC.
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
        # Keltner Channel
        self.kc_ema_period = int(config.get("kc_ema_period", 20))
        self.atr_period = int(config.get("atr_period", 14))
        self.kc_mult = float(config.get("kc_mult", 1.75))
        # Entry filters
        self.ibs_max = float(config.get("ibs_max", 0.30))
        self.consec_down_min = int(config.get("consec_down_min", 2))
        # Stop/target in ATR multiples
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
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

        # Regime filter: skip DOWN+HIGH (historically inconsistent)
        regime_trend = labels.get("regime_trend", "FLAT").upper()
        regime_vol = labels.get("regime_vol", "NORMAL").upper()
        if regime_trend == "DOWN" and regime_vol == "HIGH":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Core indicators
        ema = self._ema(closes, self.kc_ema_period)
        atr = self._atr(bars, self.atr_period)
        if ema is None or atr is None or atr < 1e-9:
            return None

        # Keltner Channel lower band
        lower_kc = ema - self.kc_mult * atr

        # Entry condition 1: price at or below lower KC
        if bar.close > lower_kc:
            return None

        # Entry condition 2: low IBS (closed near day's low)
        ibs = self._ibs(bar)
        if ibs > self.ibs_max:
            return None

        # Entry condition 3: consecutive down closes
        consec = self._consecutive_downs(closes)
        if consec < self.consec_down_min:
            return None

        # Stop and target
        stop = bar.close - self.stop_atr_mult * atr
        # In FLAT regime, target the EMA (mean reversion); otherwise fixed ATR target
        if regime_trend == "FLAT" and ema > bar.close:
            target = ema
        else:
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
                "regime": f"{regime_trend}+{regime_vol}",
                "kc_dist": round((lower_kc - bar.close) / atr, 2),
                "ibs": round(ibs, 3),
                "consec": consec,
                "seed": "keltner_ibs_regime_filter",
            },
        )
