"""Bollinger Band Squeeze Breakout — buy when volatility expands after contraction.

Entry: BB width contracts below its average (squeeze), then price breaks above upper BB.
Uses EMA trend filter + regime labels to skip DOWN markets.
Target: ATR-based. Stop: below lower BB at entry.
"""

from __future__ import annotations

import math
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
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.squeeze_lookback = int(config.get("squeeze_lookback", 20))
        self.squeeze_pctile = float(config.get("squeeze_pctile", 0.25))
        self.ema_period = int(config.get("ema_period", 20))
        self.ema_slow_period = int(config.get("ema_slow_period", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 10))

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
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = v * mult + ema * (1 - mult)
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

        # Skip near-earnings
        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings"):
            return None

        # Skip DOWN trend via regime labels
        regime_trend = regime_labels.get("regime_trend", "FLAT")
        if regime_trend == "DOWN":
            return None

        # Skip SHOCK volatility
        regime_vol = regime_labels.get("regime_vol", "NORMAL")
        if regime_vol == "SHOCK":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        if len(closes) < self.bb_period + self.squeeze_lookback:
            return None

        # Compute Bollinger Bands
        bb_mean = self._sma(closes, self.bb_period)
        bb_std_val = self._std(closes, self.bb_period)
        if bb_mean is None or bb_std_val is None or bb_mean < 1e-9 or bb_std_val < 1e-9:
            return None

        upper_bb = bb_mean + self.bb_std * bb_std_val
        lower_bb = bb_mean - self.bb_std * bb_std_val
        bb_width = (upper_bb - lower_bb) / bb_mean

        # Compute BB width history to detect squeeze
        bb_widths = []
        for i in range(self.squeeze_lookback):
            idx = len(closes) - self.squeeze_lookback + i
            if idx < self.bb_period:
                continue
            subset = closes[idx - self.bb_period + 1 : idx + 1]
            m = sum(subset) / len(subset)
            s = math.sqrt(sum((x - m) ** 2 for x in subset) / len(subset))
            if m > 1e-9:
                bb_widths.append(2 * self.bb_std * s / m)

        if len(bb_widths) < self.squeeze_lookback:
            return None

        # Check if current BB width is in the bottom percentile (squeeze)
        sorted_widths = sorted(bb_widths)
        threshold_idx = max(0, int(len(sorted_widths) * self.squeeze_pctile) - 1)
        squeeze_threshold = sorted_widths[threshold_idx]

        # Recent BB width should have been squeezed (look at 1-3 bars ago)
        recent_widths = bb_widths[-3:]
        was_squeezed = any(w <= squeeze_threshold for w in recent_widths)
        if not was_squeezed:
            return None

        # Current bar must break above upper BB (expansion)
        if bar.close <= upper_bb:
            return None

        # Trend filter: EMA(20) > EMA(50) — stock in uptrend
        ema_fast = self._ema(closes, self.ema_period)
        ema_slow = self._ema(closes, self.ema_slow_period)
        if ema_fast is None or ema_slow is None:
            return None
        if ema_fast <= ema_slow:
            return None

        # ATR for stop/target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
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
                "bb_width": round(bb_width, 4),
                "squeeze_threshold": round(squeeze_threshold, 4),
                "upper_bb": round(upper_bb, 2),
                "atr": round(atr, 4),
                "seed": "vol_breakout_evolved",
            },
        )
