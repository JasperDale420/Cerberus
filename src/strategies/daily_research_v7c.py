"""Simple Breakout + Wide Target Strategy.

Buy when price closes above recent high with ATR expansion.
Wide targets (3x ATR) let winners run — short holds lose, long holds win.
Minimal filters for maximum trade generation across all regimes.

Long-only, daily bars, max_hold_days=10.
"""

from __future__ import annotations

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
        self.min_bars = int(config.get("min_bars", 50))
        self.atr_period = int(config.get("atr_period", 14))

        # Optimizable
        self.atr_expansion = float(config.get("atr_expansion", 1.1))
        self.vol_surge_mult = float(config.get("vol_surge_mult", 1.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))

        # Structural
        self.breakout_lookback = int(config.get("breakout_lookback", 5))
        self.sma_period = int(config.get("sma_period", 10))
        self.max_hold_days = int(config.get("max_hold_days", 10))

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

    def _atr_series(self, bars: list[Bar], period: int, count: int) -> list[float]:
        result = []
        for i in range(count):
            end_idx = len(bars) - count + i + 1
            if end_idx < period + 1:
                continue
            sub = bars[:end_idx]
            val = self._atr(sub, period)
            if val is not None:
                result.append(val)
        return result

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

        # Skip earnings and FOMC
        if regime_labels.get("near_earnings") or regime_labels.get("near_fomc"):
            return None

        # Skip SHOCK and DOWN
        vol_regime = regime_labels.get("regime_vol", "NORMAL")
        if vol_regime == "SHOCK":
            return None
        trend = regime_labels.get("regime_trend", "FLAT")
        if trend == "DOWN":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Breakout: close above N-day high
        if len(bars) < self.breakout_lookback + 1:
            return None
        lookback_highs = [b.high for b in bars[-(self.breakout_lookback + 1) : -1]]
        if bar.close <= max(lookback_highs):
            return None

        # ATR expansion: current ATR > threshold * average ATR
        atr_values = self._atr_series(bars, self.atr_period, 20)
        if len(atr_values) >= 10:
            atr_avg = sum(atr_values) / len(atr_values)
            if atr_avg > 1e-9:
                atr_ratio = atr / atr_avg
                if atr_ratio < self.atr_expansion:
                    return None

        # Price above SMA (trend filter)
        sma = self._sma(closes, self.sma_period)
        if sma is not None and bar.close < sma:
            return None

        # Wide stop and target
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
            meta={"mode": "BREAKOUT", "trend": trend},
        )
