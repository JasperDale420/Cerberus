"""Keltner Channel Pullback — buy dips to lower Keltner band in uptrends.

Entry: price pulls back near lower Keltner Channel while longer-term trend intact.
Market-level filter via regime_snapshot (SPY trend/vol). Stock-level EMA alignment.
Target: EMA midline. Stop: ATR-based below entry.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, TrendRegime, VolRegime
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
        self.min_bars = int(config.get("min_bars", 25))
        self.ema_period = int(config.get("ema_period", 20))
        self.ema_slow_period = int(config.get("ema_slow_period", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.keltner_mult = float(config.get("keltner_mult", 2.0))
        self.pullback_zone = float(config.get("pullback_zone", 0.75))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 10))

    # --- Indicator helpers ---

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

        # Market-level regime filter via regime_snapshot (aligns with WFO window classification)
        snapshot = market_state.regime_snapshot
        if snapshot is not None:
            # Skip DOWN market trend
            if snapshot.trend == TrendRegime.DOWN:
                return None
            # Skip SHOCK market volatility only
            if snapshot.vol == VolRegime.SHOCK:
                return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Compute indicators
        ema_fast = self._ema(closes, self.ema_period)
        ema_slow = self._ema(closes, self.ema_slow_period)
        atr = self._atr(bars, self.atr_period)

        if ema_fast is None or ema_slow is None or atr is None or atr < 1e-9:
            return None

        # Stock-level trend confirmation: fast EMA above slow EMA
        if ema_fast <= ema_slow:
            return None

        # Keltner Channel pullback zone
        pullback_threshold = ema_fast - self.pullback_zone * self.keltner_mult * atr

        # Entry: price pulled back into lower portion of Keltner channel
        if bar.close > pullback_threshold:
            return None

        # Price must still be above a safety floor (not a crash)
        safety_floor = ema_fast - 3.0 * atr
        if bar.close < safety_floor:
            return None

        # Stop below entry, target the EMA midline
        stop = bar.close - self.stop_atr_mult * atr
        target = ema_fast  # mean reversion target

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
                "ema_fast": round(ema_fast, 2),
                "ema_slow": round(ema_slow, 2),
                "pullback_depth": round((ema_fast - bar.close) / atr, 2),
                "seed": "vol_breakout_evolved",
            },
        )
