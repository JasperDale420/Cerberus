"""Confluence: Consec Down + BB + IBS, BB Midline Target.

Buy after 2+ consecutive down closes when:
- Price below BB(20) midline (short-term oversold)
- Price above BB lower band (not in freefall)
- IBS < 0.35 (closed near low = selling exhaustion)
Target: BB midline (adaptive mean reversion target).
Stop: 1.5 ATR below entry. Skips SHOCK vol and earnings.
Long-only, daily bars, max_hold_days=5.
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
        self.consec_down_min = int(config.get("consec_down_min", 2))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std_mult = float(config.get("bb_std_mult", 2.0))
        self.ibs_max = float(config.get("ibs_max", 0.35))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))

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
        data = values[-period:]
        mean = sum(data) / period
        variance = sum((x - mean) ** 2 for x in data) / period
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
    def _consecutive_down(closes: list[float]) -> int:
        """Count consecutive down closes from the end."""
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count

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

        # Condition 1: 2+ consecutive down closes
        consec = self._consecutive_down(closes)
        if consec < self.consec_down_min:
            return None

        # Condition 2: IBS < 0.35 (closed near low = selling exhaustion)
        ibs = self._ibs(bar)
        if ibs is None or ibs > self.ibs_max:
            return None

        # Condition 3: Price below BB(20) midline (short-term oversold)
        bb_mean = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mean is None or bb_std is None or bb_std < 0.01:
            return None
        if bar.close > bb_mean:
            return None

        # Condition 4: Price above BB lower band (not in freefall)
        bb_lower = bb_mean - self.bb_std_mult * bb_std
        if bar.close < bb_lower:
            return None

        # ATR for stops
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        z_score = (bar.close - bb_mean) / bb_std if bb_std > 0 else 0.0

        stop = bar.close - self.stop_atr_mult * atr
        # Target = BB midline (adaptive mean reversion target)
        target = bb_mean

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "mode": "consec_ibs_bb_target",
                "consec_down": consec,
                "ibs": round(ibs, 3),
                "z_score": round(z_score, 2),
                "bb_mean": round(bb_mean, 2),
                "atr": round(atr, 4),
            },
        )
