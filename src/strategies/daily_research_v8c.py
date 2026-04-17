"""Consecutive-down ATR mean reversion — buy after 2+ down days below SMA.

Entry requires ALL of:
  1. 2+ consecutive lower closes
  2. IBS < threshold (closed near day's low)
  3. Price below SMA(20) by at least atr_dip_min * ATR
  4. Not in HIGH/SHOCK vol
  5. Not near earnings or FOMC

Target: SMA(20) — natural mean reversion level.
Stop: entry - stop_atr_mult * ATR.
Evolved from seed_vol_breakout — ATR-scaled risk management (vol DNA).
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
        self.consec_down_min = 2
        self.ibs_threshold = 0.3  # hardcoded — was unstable (CV=0.308), mean was 0.294
        self.atr_dip_min = 0.5  # hardcoded — was unstable (CV=0.445), mean was 0.55
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.2))
        self.max_hold_days = int(config.get("max_hold_days", 3))

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

        # Day-of-week filter: skip Wed(2), Thu(3), Sat(5) — historically negative
        dow = bar.time.weekday()
        if dow in (2, 3, 5):
            return None

        # Skip HIGH and SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol in (VolRegime.HIGH, VolRegime.SHOCK):
            return None

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None
        if labels.get("opex_week", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # 2+ consecutive down closes (3+ required in DOWN trend — more selective)
        consec = self._consecutive_downs(closes)
        trend = labels.get("regime_trend", "FLAT")
        min_consec = 3 if trend == "DOWN" else self.consec_down_min
        if consec < min_consec:
            return None

        # IBS filter: close near day's low
        bar_range = bar.high - bar.low
        if bar_range > 1e-9:
            ibs = (bar.close - bar.low) / bar_range
            if ibs > self.ibs_threshold:
                return None

        # SMA — price must be below it
        sma = self._sma(closes, self.sma_period)
        if sma is None or sma < 1e-9:
            return None
        if bar.close >= sma:
            return None

        # ATR for stop sizing and dip quality
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Dip must be at least atr_dip_min * ATR below SMA
        dip = sma - bar.close
        if dip < self.atr_dip_min * atr:
            return None

        # Target: SMA in UP/FLAT, partial (60%) in DOWN — downtrends revert less
        if trend == "DOWN":
            target = bar.close + 0.6 * dip
        else:
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
                "sma": round(sma, 2),
                "dip_atr": round(dip / atr, 2),
                "seed": "consec_atr_mr",
            },
        )
