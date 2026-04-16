"""IBS Mean Reversion — maximum diversification.

High trade volume strategy for maximum per-window consistency.
More trades per window = lower variance = higher min_pf.

Signal: IBS < threshold AND 1+ consecutive down day.
Generous criteria to maximize trade count.
Only filter: skip DOWN+HIGH vol combo (proven consistent loser).
Exclude leveraged/inverse ETFs.
Skip SHOCK vol, earnings, FOMC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy

_EXCLUDED = {"VXX", "UVXY", "SQQQ", "TQQQ", "SPXU", "SPXS", "SDOW", "LABU", "LABD"}


class dailyresearchv7dStrategy(BaseStrategy):
    name = "daily_research_v7d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 20))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.25))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.down_target_scale = float(config.get("down_target_scale", 0.7))
        self.down_stop_scale = float(config.get("down_stop_scale", 0.7))

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
    def _ibs(bar: Bar) -> Optional[float]:
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
        if symbol in _EXCLUDED:
            return None
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        regime_trend = labels.get("regime_trend", "FLAT").upper()
        regime_vol = labels.get("regime_vol", "NORMAL").upper()
        if regime_trend == "DOWN" and regime_vol == "HIGH":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        if len(closes) < self.min_bars:
            return None

        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None
        if bar.close < 5.0:
            return None
        if atr / bar.close < 0.005:
            return None

        ibs = self._ibs(bar)
        if ibs is None or ibs >= self.ibs_threshold:
            return None

        # Just 1 consecutive down day (generous for more signals)
        if len(closes) >= 2 and closes[-1] >= closes[-2]:
            return None

        stop_mult = self.stop_atr_mult
        target_mult = self.target_atr_mult
        if regime_trend == "DOWN":
            target_mult *= self.down_target_scale
            stop_mult *= self.down_stop_scale

        stop = bar.close - stop_mult * atr
        target = bar.close + target_mult * atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "ibs": round(ibs, 3),
                "regime": regime_trend,
                "vol_regime": regime_vol,
                "atr": round(atr, 4),
                "seed": "ibs_max_diversification",
            },
        )
