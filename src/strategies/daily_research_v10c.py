"""ATR-Normalized Dip Buy — evolved from vol_breakout seed.

Buy when price has dipped significantly relative to ATR from its recent high,
and IBS confirms selling exhaustion. Uses ATR to normalize entry depth
so the strategy adapts to changing volatility — maintains vol_breakout DNA.

Filters: Block DOWN trend regime, skip SHOCK vol, skip earnings/FOMC.
Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v10c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.high_lookback = int(config.get("high_lookback", 10))
        self.atr_dip_min = float(config.get("atr_dip_min", 1.5))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.4))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
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

        # --- Regime filters ---
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("regime_trend", "") == "DOWN":
            return None
        if labels.get("near_earnings", False):
            return None
        if labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)

        # ATR — our core volatility measure
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Recent high over lookback period (excluding current bar)
        if len(bars) < self.high_lookback + 1:
            return None
        recent_highs = [b.high for b in bars[-(self.high_lookback + 1) : -1]]
        recent_high = max(recent_highs)

        # ATR-normalized dip: how many ATRs has price dropped from recent high?
        dip_depth = (recent_high - bar.close) / atr
        if dip_depth < self.atr_dip_min:
            return None  # Not deep enough

        # IBS: selling exhaustion — close near day's low
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None  # Sellers haven't given up yet

        # Stop and target based on ATR
        stop = bar.close - self.stop_atr_mult * atr
        target = bar.close + self.target_atr_mult * atr

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
                "dip_depth_atr": round(dip_depth, 2),
                "recent_high": round(recent_high, 2),
                "ibs": round(ibs, 3),
                "seed": "vol_breakout",
            },
        )
