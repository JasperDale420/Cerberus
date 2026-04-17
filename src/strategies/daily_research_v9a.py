"""Volatility breakout: buy when vol compresses then expands bullishly.

Entry: ATR(5) < squeeze_ratio * ATR(14) (vol compression) +
       today's range > expansion_mult * ATR(14) (expansion) +
       close > open (bullish day).
Regime-agnostic — vol squeeze/expansion is universal.
Stop 1.5 ATR, target 3.0 ATR.
Skip SHOCK vol + earnings/FOMC.
Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v9a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.atr_fast = int(config.get("atr_fast", 5))
        self.atr_slow = int(config.get("atr_slow", 14))
        self.squeeze_ratio = float(config.get("squeeze_ratio", 0.75))
        self.expansion_mult = float(config.get("expansion_mult", 1.2))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicator helpers ---

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

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Skip earnings and FOMC
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)

        # Compute fast and slow ATR
        atr_fast = self._atr(bars, self.atr_fast)
        atr_slow = self._atr(bars, self.atr_slow)
        if atr_fast is None or atr_slow is None or atr_slow < 1e-9:
            return None

        # Vol compression: fast ATR much lower than slow ATR
        if atr_fast >= self.squeeze_ratio * atr_slow:
            return None  # No squeeze

        # Vol expansion: today's range exceeds slow ATR
        today_range = bar.high - bar.low
        if today_range < self.expansion_mult * atr_slow:
            return None  # No expansion

        # Bullish close: close above open
        if bar.close <= bar.open:
            return None

        stop = bar.close - self.stop_atr_mult * atr_slow
        target = bar.close + self.target_atr_mult * atr_slow

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "atr_fast": round(atr_fast, 4),
                "atr_slow": round(atr_slow, 4),
                "squeeze_ratio": round(atr_fast / atr_slow, 3),
                "expansion": round(today_range / atr_slow, 3),
                "seed": "mean_reversion",
            },
        )
