"""Consecutive Down + BB Proximity + IBS mean reversion.

Entry requires ALL of:
  1. N+ consecutive lower closes (consec_down_min, optimized 2-3)
  2. Low IBS (closed near day's low, optimized 0.20-0.35)
  3. Close near lower BB (within bb_proximity std devs, optimized 0.5-1.5)

Skip SHOCK vol, earnings, FOMC, opex week.
In DOWN regime, require consec >= 3 regardless of consec_down_min.
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
        self.consec_down_min = int(config.get("consec_down_min", 2))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.25))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        # Fixed — not optimized to avoid CV instability
        self.bb_proximity = 1.0

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

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None
        if labels.get("opex_week", False):
            return None

        regime_trend = labels.get("regime_trend", "FLAT").upper()

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Required: consecutive down closes
        consec = self._consecutive_downs(closes)
        required_consec = self.consec_down_min
        # In DOWN regime, require at least 3 consecutive downs
        if regime_trend == "DOWN":
            required_consec = max(required_consec, 3)
        if consec < required_consec:
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
        # In UP trend, require closer to/below lower BB (stronger signal needed)
        effective_proximity = 0.5 if regime_trend == "UP" else self.bb_proximity
        if bb_distance > effective_proximity:
            return None

        # ATR for stop and target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        # Target: min of BB midline or ATR-based target
        target_bb = bb_sma
        target_atr = bar.close + self.target_atr_mult * atr
        target = min(target_bb, target_atr)

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
                "regime_trend": regime_trend,
                "atr": round(atr, 4),
                "seed": "consec_ibs_bb",
            },
        )
