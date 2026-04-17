"""Volatility-adjusted mean reversion — ATR-based dip buying.

Entry requires ALL of:
  1. Price dropped > atr_drop_mult * ATR below SMA(20) (significant dip)
  2. Volume spike > vol_min_ratio * avg volume (exhaustion selling)
  3. Not in SHOCK volatility regime
  4. Not near earnings or FOMC

Target: SMA(20) — natural mean reversion level.
Stop: entry - stop_atr_mult * ATR.
Evolved from seed_vol_breakout — uses ATR/volume DNA for mean reversion.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v8c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.sma_period = int(config.get("sma_period", 20))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_drop_mult = float(config.get("atr_drop_mult", 1.5))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 1.2))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
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

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # SMA
        sma = self._sma(closes, self.sma_period)
        if sma is None or sma < 1e-9:
            return None

        # ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Entry: price dropped significantly below SMA (ATR-scaled dip)
        dip = sma - bar.close
        if dip < self.atr_drop_mult * atr:
            return None

        # Volume confirmation: selling exhaustion (above-average volume)
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # Target: SMA (mean reversion)
        target = sma
        if target <= bar.close:
            return None

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
                "atr": round(atr, 4),
                "sma": round(sma, 2),
                "dip_atr": round(dip / atr, 2),
                "vol_ratio": round(bar.volume / avg_vol, 2),
                "seed": "vol_mean_reversion",
            },
        )
