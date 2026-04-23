"""ATR-Envelope Dip Buy — evolved from vol_breakout seed.

Buy when close dips below SMA(20) by at least atr_dip_min * ATR,
and IBS confirms selling exhaustion. Target: SMA(20) mean reversion.
Uses ATR for envelope, stop, and normalization — vol_breakout DNA.

Filters: Skip SHOCK vol, skip earnings/FOMC.
Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v10c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.sma_period = int(config.get("sma_period", 20))
        self.atr_dip_min = float(config.get("atr_dip_min", 0.5))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.4))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicator helpers ---

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

        # --- Regime filters ---
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False):
            return None
        if labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # SMA — our mean reversion anchor
        sma = self._sma(closes, self.sma_period)
        if sma is None:
            return None

        # ATR — volatility normalizer
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # ATR-envelope dip: close must be below SMA by at least atr_dip_min * ATR
        lower_band = sma - self.atr_dip_min * atr
        if bar.close >= lower_band:
            return None

        # IBS: selling exhaustion — close near day's low
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # Stop: ATR below entry, Target: SMA (mean reversion)
        stop = bar.close - self.stop_atr_mult * atr
        target = sma

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
                "sma": round(sma, 2),
                "dip_atr": round((sma - bar.close) / atr, 2),
                "ibs": round(ibs, 3),
                "seed": "vol_breakout",
            },
        )
