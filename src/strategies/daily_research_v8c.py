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
        self.min_bars = int(config.get("min_bars", 55))
        self.sma_fast = int(config.get("sma_fast", 20))
        self.sma_slow = int(config.get("sma_slow", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_drop_mult = float(config.get("atr_drop_mult", 1.2))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 1.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.40))
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

        # Skip HIGH and SHOCK volatility (HIGH vol windows consistently lose)
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol in (VolRegime.HIGH, VolRegime.SHOCK):
            return None

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # SMA fast (mean reversion anchor) and slow (trend filter)
        sma_fast = self._sma(closes, self.sma_fast)
        sma_slow = self._sma(closes, self.sma_slow)
        if sma_fast is None or sma_slow is None or sma_fast < 1e-9:
            return None

        # Uptrend filter: price must be above SMA(50) to avoid falling knives
        if bar.close < sma_slow:
            return None

        # ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # IBS filter: close near day's low (better mean reversion timing)
        bar_range = bar.high - bar.low
        if bar_range > 1e-9:
            ibs = (bar.close - bar.low) / bar_range
            if ibs > self.ibs_threshold:
                return None

        # Entry: price pulled back below SMA(20) by at least atr_drop_mult * ATR
        dip = sma_fast - bar.close
        if dip < self.atr_drop_mult * atr:
            return None

        # Volume confirmation: above-average volume (selling exhaustion)
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # Target: SMA fast (mean reversion)
        target = sma_fast
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
                "sma_fast": round(sma_fast, 2),
                "sma_slow": round(sma_slow, 2),
                "dip_atr": round(dip / atr, 2),
                "vol_ratio": round(bar.volume / avg_vol, 2) if avg_vol else 0,
                "ibs": round((bar.close - bar.low) / bar_range, 3) if bar_range > 1e-9 else 0.5,
                "seed": "vol_mean_reversion",
            },
        )
