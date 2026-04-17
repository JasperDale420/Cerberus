"""Keltner-IBS Reversion: Buy at Keltner Lower Band with IBS Confirmation.

Entry: Close below Keltner lower band (EMA20 - 1.5*ATR) AND IBS < 0.3.
Trend filter: EMA(20) > EMA(50) ensures we're buying dips in uptrends.
Target: EMA(20) (natural mean reversion level).
Stop: 2.0 ATR below entry.
Skips SHOCK vol and earnings. Long-only, daily bars, max_hold_days=5.
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
        self.ema_fast_period = int(config.get("ema_fast_period", 20))
        self.ema_slow_period = int(config.get("ema_slow_period", 50))
        self.keltner_atr_mult = float(config.get("keltner_atr_mult", 1.5))
        self.ibs_max = float(config.get("ibs_max", 0.3))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))

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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Trend filter: EMA(20) > EMA(50) — only buy dips in uptrends
        ema_fast = self._ema(closes, self.ema_fast_period)
        ema_slow = self._ema(closes, self.ema_slow_period)
        if ema_fast is None or ema_slow is None:
            return None
        if ema_fast <= ema_slow:
            return None

        # ATR for Keltner bands and stops
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Keltner lower band: EMA(20) - mult * ATR
        keltner_lower = ema_fast - self.keltner_atr_mult * atr

        # Entry: close at or below Keltner lower band
        if bar.close > keltner_lower:
            return None

        # IBS confirmation: close near the low (selling exhaustion)
        ibs = self._ibs(bar)
        if ibs is None or ibs > self.ibs_max:
            return None

        # Don't buy too far below (broken support, >3 ATR below EMA)
        if bar.close < ema_fast - 3.0 * atr:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        # Target: EMA(20) — natural mean reversion level
        target = ema_fast

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "keltner_ibs",
                "ema_fast": round(ema_fast, 2),
                "ema_slow": round(ema_slow, 2),
                "keltner_lower": round(keltner_lower, 2),
                "ibs": round(ibs, 3),
                "atr": round(atr, 4),
            },
        )
