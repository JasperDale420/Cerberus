"""Daily Research v10b — BB + IBS + Consecutive Down Confluence.

Buy when price closes below lower Bollinger Band, IBS is low (close near day low),
and there have been 2+ consecutive down days. Minimal filters for trade count.
Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v10b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 25))
        # Locked params (from param_spaces.py)
        self.atr_period = int(config.get("atr_period", 14))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        # Tunable params (optimized by Optuna)
        self.ibs_threshold = float(config.get("ibs_threshold", 0.25))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        # Fixed internal params
        self.consec_down_min = int(config.get("consec_down_min", 2))
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
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
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
    def _count_consecutive_down(closes: list[float]) -> int:
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

        # Skip SHOCK and HIGH volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol in (VolRegime.SHOCK, VolRegime.HIGH):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Bollinger Bands
        bb_mean = self._sma(closes, self.bb_period)
        bb_sd = self._std(closes, self.bb_period)
        if bb_mean is None or bb_sd is None or bb_sd < 1e-9:
            return None

        lower_band = bb_mean - self.bb_std * bb_sd

        # Price must be at or below lower Bollinger Band
        if bar.close > lower_band:
            return None

        # IBS: close near the low of the day
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs > self.ibs_threshold:
            return None

        # Consecutive down days
        consec_down = self._count_consecutive_down(closes)
        if consec_down < self.consec_down_min:
            return None

        # ATR for stop and target
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Skip earnings
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False):
            return None

        stop = bar.close - self.stop_atr_mult * atr
        # Target = BB mean (natural mean-reversion level), floored by ATR minimum
        target_bb = bb_mean
        target_atr = bar.close + self.target_atr_mult * atr
        target = max(target_bb, bar.close + 0.5 * atr)  # at least 0.5 ATR
        if target_bb < target_atr:
            target = target_bb  # prefer BB mean when it's closer

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "bb_lower": round(lower_band, 2),
                "bb_mean": round(bb_mean, 2),
                "ibs": round(ibs, 3),
                "consec_down": consec_down,
                "atr": round(atr, 4),
                "seed": "trend_pullback",
            },
        )
