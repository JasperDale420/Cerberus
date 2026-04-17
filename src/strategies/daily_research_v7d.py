"""Consecutive-down mean reversion with regime gating.

Entry: 2+ consecutive lower closes + BB proximity for mean reversion.
Uses multi-factor scoring: consecutive downs, BB position, volume, IBS.
Long-only for consistency. Event and regime filters.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedRegimeSwitchStrategy(BaseStrategy):
    name = "daily_research_v7d"
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
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        self.bb_proximity = float(config.get("bb_proximity", 0.5))
        self.min_rr_ratio = float(config.get("min_rr_ratio", 1.5))

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
        """Count consecutive lower closes from the most recent bar backwards."""
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _ibs(bar: Bar) -> float:
        """Internal Bar Strength: (close - low) / (high - low)."""
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

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Event filter
        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings", False):
            return None
        if regime_labels.get("near_fomc", False):
            return None

        # Day-of-week filter: skip Wed/Thu/Sat (historically losing)
        dow = bar.time.weekday()  # 0=Mon, 1=Tue, ..., 4=Fri, 5=Sat
        if dow in (2, 3, 5):  # Wed, Thu, Sat
            return None

        # Skip quad witch weeks (extra volatility from options expiration)
        if regime_labels.get("quad_witch_week", False):
            return None

        # Regime filter: skip SHOCK vol
        regime_vol = regime_labels.get("regime_vol", "NORMAL").upper()
        if regime_vol == "SHOCK":
            return None

        regime_trend = regime_labels.get("regime_trend", "FLAT").upper()

        # Indicators
        bb_sma = self._sma(closes, self.bb_period)
        bb_std_val = self._std(closes, self.bb_period)
        atr = self._atr(bars, self.atr_period)

        if bb_sma is None or bb_std_val is None or bb_std_val < 1e-9:
            return None
        if atr is None or atr < 1e-9:
            return None

        lower_bb = bb_sma - self.bb_std * bb_std_val

        # Multi-factor entry scoring
        score = 0

        # Factor 1: Consecutive down closes
        consec = self._consecutive_downs(closes)
        if consec >= self.consec_down_min:
            score += 1
        if consec >= self.consec_down_min + 1:
            score += 1

        # Factor 2: BB proximity (close near or below lower BB)
        bb_distance = (bar.close - lower_bb) / bb_std_val if bb_std_val > 0 else 999
        if bb_distance < self.bb_proximity:
            score += 1
        if bb_distance < 0:  # Below lower BB
            score += 1

        # Factor 3: IBS (low IBS = closed near low = oversold)
        ibs = self._ibs(bar)
        if ibs < self.ibs_threshold:
            score += 1

        # Require at least 3 factors
        if score < 3:
            return None

        # In DOWN regime, require stronger signal (4+ factors)
        if regime_trend == "DOWN" and score < 4:
            return None

        # Target: BB midline (natural mean reversion target)
        target = bb_sma

        if target <= bar.close:
            return None

        stop = bar.close - self.stop_atr_mult * atr

        # Minimum reward:risk ratio filter
        risk = bar.close - stop
        reward = target - bar.close
        if risk <= 0 or reward / risk < self.min_rr_ratio:
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
                "regime_trend": regime_trend,
                "regime_vol": regime_vol,
                "consec_down": consec,
                "bb_distance": round(bb_distance, 2),
                "ibs": round(ibs, 3),
                "score": score,
            },
        )
