"""Multi-factor mean reversion with regime gating.

Entry: Confluence of short-term pullback (3-day ROC), BB lower band,
and volume confirmation. Long-only for consistency.
Regime-adaptive: skip DOWN+SHOCK, tighter in DOWN regimes.
Event filter: skip earnings and FOMC days.
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
        self.min_bars = int(config.get("min_bars", 55))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.roc_period = int(config.get("roc_period", 3))
        self.roc_threshold = float(config.get("roc_threshold", -0.03))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_avg_mult = float(config.get("vol_avg_mult", 0.5))

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

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
        volumes = [b.volume for b in bars]

        # Event filter: skip earnings and FOMC
        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings", False):
            return None
        if regime_labels.get("near_fomc", False):
            return None

        # Regime filter: skip SHOCK vol
        regime_vol = regime_labels.get("regime_vol", "NORMAL").upper()
        if regime_vol == "SHOCK":
            return None

        regime_trend = regime_labels.get("regime_trend", "FLAT").upper()

        # Bollinger Bands
        bb_sma = self._sma(closes, self.bb_period)
        bb_std_val = self._std(closes, self.bb_period)
        if bb_sma is None or bb_std_val is None or bb_std_val < 1e-9:
            return None

        lower_bb = bb_sma - self.bb_std * bb_std_val
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # 3-day rate of change
        if len(closes) < self.roc_period + 1:
            return None
        roc = (closes[-1] - closes[-1 - self.roc_period]) / closes[-1 - self.roc_period]

        # Volume check
        avg_vol = self._sma(volumes[-self.vol_avg_period :], min(self.vol_avg_period, len(volumes)))
        if avg_vol is None or avg_vol < 1:
            return None
        current_vol = volumes[-1]
        if current_vol < avg_vol * self.vol_avg_mult:
            return None

        # LONG signal: price below BB lower + short-term pullback
        if bar.close < lower_bb and roc < self.roc_threshold:
            # In DOWN regime, require deeper pullback
            if regime_trend == "DOWN" and roc > self.roc_threshold * 1.5:
                return None

            stop = bar.close - self.stop_atr_mult * atr
            target = bb_sma  # Mean reversion to BB midline

            if target <= bar.close:
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
                    "regime_trend": regime_trend,
                    "regime_vol": regime_vol,
                    "roc3": round(roc, 4),
                    "bb_lower": round(lower_bb, 2),
                    "bb_mid": round(bb_sma, 2),
                    "atr": round(atr, 2),
                },
            )

        return None
