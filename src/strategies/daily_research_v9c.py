"""Volatility Squeeze Breakout — Buy after vol compression resolves upward.

Evolved from seed_vol_breakout. The key insight: vol compression (Bollinger Bands
inside Keltner Channel) followed by expansion creates reliable breakouts.

Entry:
1. Squeeze detected: BB width < Keltner width (vol compressed)
2. Squeeze releases: BB expands back outside Keltner
3. Momentum positive: close > close[N] (price moving up during release)
4. Trend filter: close > SMA(20)
5. Skip earnings, FOMC, SHOCK vol

Exit: ATR-based stop and target. Max hold 7 days.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v9c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True
        self._squeeze_count: dict[str, int] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 30))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_mult = float(config.get("bb_mult", 2.0))
        self.kc_period = int(config.get("kc_period", 20))
        self.kc_mult = float(config.get("kc_mult", 1.5))
        self.atr_period = int(config.get("atr_period", 14))
        self.mom_lookback = int(config.get("mom_lookback", 12))
        self.sma_period = int(config.get("sma_period", 20))
        self.min_squeeze_bars = int(config.get("min_squeeze_bars", 3))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 7))

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
        return variance**0.5

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
        if regime_labels.get("near_fomc"):
            return None

        regime_vol = regime_labels.get("regime_vol", "NORMAL")
        if regime_vol == "SHOCK":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        if len(closes) < max(self.bb_period, self.kc_period, self.mom_lookback) + 1:
            return None

        # ATR for Keltner Channel
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Bollinger Bands
        bb_mid = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mid is None or bb_std is None or bb_std < 1e-9:
            return None

        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std
        bb_width = bb_upper - bb_lower

        # Keltner Channel
        kc_mid = self._sma(closes, self.kc_period)
        if kc_mid is None:
            return None
        kc_upper = kc_mid + self.kc_mult * atr
        kc_lower = kc_mid - self.kc_mult * atr
        kc_width = kc_upper - kc_lower

        # Squeeze detection: BB inside Keltner
        in_squeeze = bb_width < kc_width

        if symbol not in self._squeeze_count:
            self._squeeze_count[symbol] = 0

        if in_squeeze:
            self._squeeze_count[symbol] += 1
            return None  # Still in squeeze - wait

        # Squeeze just released (was in squeeze, now not)
        squeeze_bars = self._squeeze_count[symbol]
        self._squeeze_count[symbol] = 0

        if squeeze_bars < self.min_squeeze_bars:
            return None  # Not enough compression

        # Momentum: close > close[N bars ago] (upward resolution)
        if len(closes) < self.mom_lookback + 1:
            return None
        if bar.close <= closes[-self.mom_lookback - 1]:
            return None

        # Trend filter: price above SMA
        sma = self._sma(closes, self.sma_period)
        if sma is None or bar.close <= sma:
            return None

        # Stop and target
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
                "atr": round(atr, 4),
                "squeeze_bars": squeeze_bars,
                "bb_width": round(bb_width, 4),
                "kc_width": round(kc_width, 4),
                "momentum": round(bar.close - closes[-self.mom_lookback - 1], 2),
                "regime_vol": regime_vol,
                "seed": "vol_squeeze_v1",
            },
        )
