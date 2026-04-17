"""Consecutive-down + BB + ATR mean reversion.

Entry requires ALL of:
  1. 2+ consecutive lower closes
  2. IBS < threshold (closed near day's low)
  3. Close near lower Bollinger Band (within bb_proximity std devs)
  4. Price below SMA(20)
  5. Dip at least atr_dip_min * ATR below SMA
  6. Not in HIGH/SHOCK vol
  7. Not near earnings or FOMC

Target: BB midline (SMA). Stop: ATR-scaled below entry.
Evolved from seed_vol_breakout — ATR + BB volatility-based approach.
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
        self.bb_std = float(config.get("bb_std", 2.0))
        self.bb_proximity = float(config.get("bb_proximity", 1.2))
        self.atr_period = int(config.get("atr_period", 14))
        self.consec_down_min = 2
        self.ibs_threshold = float(config.get("ibs_threshold", 0.40))
        self.atr_dip_min = float(config.get("atr_dip_min", 0.5))
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

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        subset = values[-period:]
        mean = sum(subset) / period
        variance = sum((v - mean) ** 2 for v in subset) / period
        return variance**0.5

    @staticmethod
    def _consecutive_downs(closes: list[float]) -> int:
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count

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

        # 2+ consecutive down closes
        consec = self._consecutive_downs(closes)
        if consec < self.consec_down_min:
            return None

        # IBS filter: close near day's low
        bar_range = bar.high - bar.low
        if bar_range > 1e-9:
            ibs = (bar.close - bar.low) / bar_range
            if ibs > self.ibs_threshold:
                return None

        # SMA and BB
        sma = self._sma(closes, self.sma_period)
        std_val = self._std(closes, self.sma_period)
        if sma is None or std_val is None or sma < 1e-9 or std_val < 1e-9:
            return None

        # Price must be below SMA
        if bar.close >= sma:
            return None

        # BB proximity: close must be near lower BB
        lower_bb = sma - self.bb_std * std_val
        bb_distance = (bar.close - lower_bb) / std_val
        if bb_distance > self.bb_proximity:
            return None

        # ATR for stop sizing and dip quality
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Dip must be at least atr_dip_min * ATR below SMA
        dip = sma - bar.close
        if dip < self.atr_dip_min * atr:
            return None

        # Target: BB midline (SMA)
        target = sma
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
                "consec_down": consec,
                "atr": round(atr, 4),
                "bb_dist": round(bb_distance, 2),
                "dip_atr": round(dip / atr, 2),
                "seed": "consec_bb_atr",
            },
        )
