"""Regime-Adaptive Confluence: Consecutive Down Days + BB + IBS.

UP/FLAT: Buy after 2+ consecutive down closes below BB midline.
DOWN: Require 3+ consecutive down days AND IBS < 0.2 (stronger oversold).
Skips SHOCK vol, HIGH vol in DOWN regime, and earnings.
Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v9b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.consec_down_min = int(config.get("consec_down_min", 2))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.sma_long_period = int(config.get("sma_long_period", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.8))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        data = values[-period:]
        mean = sum(data) / period
        variance = sum((x - mean) ** 2 for x in data) / period
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
    def _consecutive_down(closes: list[float]) -> int:
        """Count consecutive down closes from the end."""
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _ibs(bar: Bar) -> Optional[float]:
        """Internal Bar Strength: (close - low) / (high - low)."""
        rng = bar.high - bar.low
        if rng < 1e-9:
            return None
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

        # Skip near earnings
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False):
            return None

        # Get regime trend
        trend = labels.get("regime_trend", "FLAT")

        # In DOWN + HIGH vol: skip entirely (catching falling knives)
        if trend == "DOWN" and snapshot and snapshot.vol == VolRegime.HIGH:
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # Condition 1: Consecutive down closes (regime-adaptive threshold)
        consec = self._consecutive_down(closes)
        min_consec = self.consec_down_min if trend != "DOWN" else self.consec_down_min + 1
        if consec < min_consec:
            return None

        # In DOWN regime, also require low IBS for better entry timing
        if trend == "DOWN":
            ibs = self._ibs(bar)
            if ibs is None or ibs > 0.2:
                return None

        # Condition 2: Price in lower half of Bollinger Band
        bb_mean = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mean is None or bb_std is None or bb_std < 0.01:
            return None

        if bar.close > bb_mean:
            return None

        # Condition 3: Not in total freefall — price above SMA(50) - 2*ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        sma_long = self._sma(closes, self.sma_long_period)
        if sma_long is not None and bar.close < sma_long - 2.0 * atr:
            return None

        # Condition 4: Volume above average
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        z_score = (bar.close - bb_mean) / bb_std if bb_std > 0 else 0.0

        # Tighter stops in DOWN regime
        stop_mult = self.stop_atr_mult if trend != "DOWN" else self.stop_atr_mult * 0.8
        target_mult = self.target_atr_mult if trend != "DOWN" else self.target_atr_mult * 0.7

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
                "mode": "consec_down_bb",
                "consec_down": consec,
                "z_score": round(z_score, 2),
                "trend": trend,
                "atr": round(atr, 4),
            },
        )
