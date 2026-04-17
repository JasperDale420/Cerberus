"""ATR-scaled mean reversion — buy dips below SMA with vol exhaustion.

Entry requires ALL of:
  1. Price below SMA(20) by at least atr_drop_mult * ATR
  2. IBS < threshold (closed near day's low — selling exhaustion)
  3. Not in HIGH/SHOCK vol regime
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
        self.atr_drop_mult = float(config.get("atr_drop_mult", 0.8))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.40))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))

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

        # Skip HIGH and SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol in (VolRegime.HIGH, VolRegime.SHOCK):
            return None

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        sma = self._sma(closes, self.sma_period)
        if sma is None or sma < 1e-9:
            return None

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # IBS filter: close near day's low (selling exhaustion)
        bar_range = bar.high - bar.low
        if bar_range > 1e-9:
            ibs = (bar.close - bar.low) / bar_range
            if ibs > self.ibs_threshold:
                return None

        # Entry: price below SMA by at least atr_drop_mult * ATR
        dip = sma - bar.close
        if dip < self.atr_drop_mult * atr:
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
                "ibs": round((bar.close - bar.low) / bar_range, 3) if bar_range > 1e-9 else 0.5,
                "seed": "vol_mean_rev",
            },
        )
