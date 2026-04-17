"""Vol-Filtered Oversold Bounce (vol_breakout archetype).

Buy after consecutive down days when ATR is elevated (vol expansion).
The idea: volatility expansion creates oversold bounces that are more
reliable than in calm markets. Uses IBS (close position) and consecutive
down-day count as entry, ATR ratio as vol filter.

NOT a pure RSI strategy — entry is structural (down days + IBS).
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
        self.min_bars = int(config.get("min_bars", 30))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.min_consec_down = int(config.get("min_consec_down", 2))
        self.atr_expansion = float(config.get("atr_expansion", 1.1))

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
        if regime_labels.get("regime_vol") == "SHOCK":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        if len(bars) < 25:
            return None

        atr = self._atr(bars, 14)
        if atr is None or atr < 1e-9:
            return None

        # Consecutive down days (close < prior close)
        consec_down = 0
        for i in range(len(bars) - 1, 0, -1):
            if bars[i].close < bars[i - 1].close:
                consec_down += 1
            else:
                break
        if consec_down < self.min_consec_down:
            return None

        # IBS: close near the low of the day (oversold signal)
        today_range = bar.high - bar.low
        if today_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / today_range
        if ibs > 0.35:
            return None

        # ATR expansion filter: current ATR vs longer-term ATR
        atr_long = self._atr(bars[:-5], 20)
        if atr_long is None or atr_long < 1e-9:
            return None
        atr_ratio = atr / atr_long
        if atr_ratio < self.atr_expansion:
            return None

        # Must be above SMA(50) — not in deep downtrend
        sma50 = self._sma(closes, 50)
        if sma50 is not None and bar.close < sma50 * 0.95:
            return None

        # Regime-adaptive stops
        regime_trend = regime_labels.get("regime_trend", "UP")
        regime_vol = regime_labels.get("regime_vol", "NORMAL")
        if regime_trend == "DOWN" or regime_vol == "HIGH":
            stop_mult = self.stop_atr_mult * 0.8
            target_mult = self.target_atr_mult * 0.7
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
                "atr": round(atr, 4),
                "atr_ratio": round(atr_ratio, 2),
                "consec_down": consec_down,
                "ibs": round(ibs, 2),
                "seed": "vol_breakout",
            },
        )
