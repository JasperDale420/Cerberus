"""Consecutive Down + IBS + BB mean reversion — strict 3-day decline variant.

Entry requires ALL of:
  1. 3+ consecutive lower closes (fixed, not optimized)
  2. Low IBS (closed near day's low, optimized 0.20-0.35)
  3. Close within 1.0 std dev of lower BB

3 consecutive downs is a stronger signal than 2 — reduces false entries
in choppy markets. Target = BB midline (natural mean reversion target).
Skip SHOCK vol, earnings, FOMC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v8b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.consec_down_min = 3  # Fixed — 3 is stronger signal than 2
        self.ibs_threshold = float(config.get("ibs_threshold", 0.25))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        self.bb_proximity = 1.0  # Fixed
        self.target_pct = float(config.get("target_pct", 0.7))  # % of distance to BB mid

    # --- Indicator helpers ---

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
    def _consecutive_downs(closes: list[float]) -> int:
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _ibs(bar: Bar) -> float:
        rng = bar.high - bar.low
        if rng < 1e-9:
            return 0.5
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

        # Skip SHOCK volatility and DOWN+HIGH regime (consistent losers)
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None
        labels = symbol_state.meta.get("regime_labels", {})
        regime_trend = labels.get("regime_trend", "")
        regime_vol = labels.get("regime_vol", "")
        if regime_trend == "DOWN" and regime_vol == "HIGH":
            return None

        # Event filters
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Required: 3+ consecutive down closes
        consec = self._consecutive_downs(closes)
        if consec < self.consec_down_min:
            return None

        # Required: Low IBS (closed near day's low)
        ibs = self._ibs(bar)
        if ibs > self.ibs_threshold:
            return None

        # Required: Close near lower BB
        bb_sma = self._sma(closes, self.bb_period)
        bb_std_val = self._std(closes, self.bb_period)
        if bb_sma is None or bb_std_val is None or bb_std_val < 1e-9:
            return None

        lower_bb = bb_sma - self.bb_std * bb_std_val
        bb_distance = (bar.close - lower_bb) / bb_std_val
        if bb_distance > self.bb_proximity:
            return None

        # ATR for stop
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Widen stop in HIGH vol to avoid premature stopouts
        stop_mult = self.stop_atr_mult
        if regime_vol == "HIGH":
            stop_mult *= 1.3
        stop = bar.close - stop_mult * atr
        # Target: partial distance to BB midline (take profit sooner)
        target = bar.close + self.target_pct * (bb_sma - bar.close)

        if target <= bar.close:
            return None

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
                "ibs": round(ibs, 3),
                "bb_distance": round(bb_distance, 2),
                "atr": round(atr, 4),
                "seed": "consec3_ibs_bb",
            },
        )
