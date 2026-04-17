"""Regime-adaptive oversold bounce with minimal parameters.

Entry: 3+ consecutive lower closes AND low IBS (closed near low of day).
Simple two-condition filter — fewer params = less overfitting.
Regime switch: only skip SHOCK vol. Stock-level SMA(50) trend filter.
In DOWN regime, require deeper oversold (stricter IBS).
Long-only. Event filters.
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
        self.consec_down_min = int(config.get("consec_down_min", 3))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.25))
        self.sma_trend_period = int(config.get("sma_trend_period", 50))

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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Event filter
        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings", False):
            return None
        if regime_labels.get("near_fomc", False):
            return None

        # Regime filter: skip SHOCK and HIGH vol (too inconsistent)
        regime_vol = regime_labels.get("regime_vol", "NORMAL").upper()
        if regime_vol in ("SHOCK", "HIGH"):
            return None
        regime_trend = regime_labels.get("regime_trend", "FLAT").upper()

        # Stock-level trend: close > SMA(50)
        sma50 = self._sma(closes, self.sma_trend_period)
        if sma50 is not None and bar.close < sma50:
            return None

        # Condition 1: Consecutive lower closes
        consec = self._consecutive_downs(closes)
        if consec < self.consec_down_min:
            return None

        # Condition 2: Low IBS (closed near low)
        ibs = self._ibs(bar)
        # In DOWN regime, require deeper oversold for safety
        if regime_trend == "DOWN":
            ibs_thresh = 0.15
        elif regime_vol == "HIGH":
            ibs_thresh = 0.2
        else:
            ibs_thresh = self.ibs_threshold
        if ibs > ibs_thresh:
            return None

        # ATR for stops/targets
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        target = bar.close + self.target_atr_mult * atr
        stop = bar.close - self.stop_atr_mult * atr

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
                "consec_down": consec,
                "ibs": round(ibs, 3),
            },
        )
