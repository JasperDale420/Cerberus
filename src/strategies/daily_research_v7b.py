"""Regime-Adaptive IBS + Multi-Factor Strategy.

Uses Internal Bar Strength (IBS) mean reversion as the core signal,
with regime-adaptive thresholds and multiple orthogonal confirmation factors.

Entry logic:
  - IBS < threshold (close near day's low = oversold intraday)
  - At least 1-4 consecutive down days (selling pressure, not free-fall)
  - Volume above minimum (participation)
  - Regime adapts IBS threshold and stop/target sizing

Regime adaptation:
  - UP: moderate IBS threshold, price must be above SMA
  - DOWN: stricter IBS threshold (deeper oversold required)
  - FLAT: standard IBS threshold (most MR-friendly)

Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v7b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        # IBS thresholds
        self.ibs_buy_threshold = float(config.get("ibs_buy_threshold", 0.25))
        # Trend filter
        self.sma_period = int(config.get("sma_period", 40))
        # Volatility
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 7))
        # Volume filter
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.7))
        # Range filter
        self.min_range_atr_ratio = float(config.get("min_range_atr_ratio", 0.5))
        # Down regime adjustment
        self.ibs_down_threshold = float(config.get("ibs_down_threshold", 0.15))
        # Consecutive down days
        self.min_down_days = int(config.get("min_down_days", 1))
        self.max_down_days = int(config.get("max_down_days", 4))

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

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
    def _ibs(bar: Bar) -> Optional[float]:
        rng = bar.high - bar.low
        if rng < 1e-9:
            return None
        return (bar.close - bar.low) / rng

    @staticmethod
    def _count_down_days(bars: list[Bar], max_lookback: int = 5) -> int:
        count = 0
        for i in range(len(bars) - 1, 0, -1):
            if bars[i].close < bars[i - 1].close:
                count += 1
            else:
                break
            if count >= max_lookback:
                break
        return count

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

        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        ibs = self._ibs(bar)
        if ibs is None:
            return None

        # Range filter: skip tiny-range days
        day_range = bar.high - bar.low
        if day_range < self.min_range_atr_ratio * atr:
            return None

        # Volume confirmation
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # Trend context
        sma = self._sma(closes, self.sma_period)
        if sma is None:
            return None

        labels = symbol_state.meta.get("regime_labels", {})
        trend = labels.get("regime_trend", "FLAT")

        # Consecutive down days filter
        down_days = self._count_down_days(bars)
        if down_days < self.min_down_days or down_days > self.max_down_days:
            return None

        # Regime-adaptive IBS threshold
        if trend == "UP":
            if ibs > self.ibs_buy_threshold:
                return None
            if bar.close < sma:
                return None
        elif trend == "DOWN":
            if ibs > self.ibs_down_threshold:
                return None
        else:
            if ibs > self.ibs_buy_threshold:
                return None

        # Dynamic stop/target by regime
        if trend == "DOWN":
            stop_mult = self.stop_atr_mult * 0.8
            target_mult = self.target_atr_mult * 0.7
        elif trend == "UP":
            stop_mult = self.stop_atr_mult
            target_mult = self.target_atr_mult * 1.2
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
                "ibs": round(ibs, 3),
                "trend": trend,
                "down_days": down_days,
                "atr": round(atr, 4),
                "vol_ratio": round(bar.volume / avg_vol, 2),
                "archetype": "regime_adaptive_ibs",
            },
        )
