"""Iteration 3: IBS + Consecutive Down Days with vol/day filters.

Buy after 2+ consecutive down closes when IBS is low.
Skip HIGH/SHOCK vol regimes. Skip Monday entries.
ATR-based stops/targets with wider target (asymmetric R:R).
Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v7a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        # IBS threshold
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        # Consecutive down days required
        self.min_down_days = int(config.get("min_down_days", 2))
        # Drawdown filter
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.15))
        # Stop/target in ATR multiples
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
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
        highs = [b.high for b in bars]

        # --- Regime filter: skip HIGH and SHOCK vol ---
        labels = symbol_state.meta.get("regime_labels", {})
        vol = labels.get("regime_vol", "")
        if vol in ("HIGH", "SHOCK"):
            return None

        # --- Day-of-week filter: skip Monday ---
        bar_time = bar.time
        if isinstance(bar_time, datetime):
            if bar_time.weekday() == 0:  # Monday
                return None

        # --- IBS filter: close must be near the low of the day ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # --- Consecutive down days ---
        if len(closes) < self.min_down_days + 1:
            return None
        for i in range(1, self.min_down_days + 1):
            if closes[-i] >= closes[-i - 1]:
                return None

        # --- Drawdown filter: skip if in freefall ---
        lookback_highs = highs[-self.drawdown_lookback :]
        peak = max(lookback_highs)
        dd = (peak - bar.close) / peak if peak > 0 else 0
        if dd > self.max_drawdown_pct:
            return None

        # --- ATR for stop/target ---
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

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
                "ibs": round(ibs, 3),
                "down_days": self.min_down_days,
                "atr": round(atr, 4),
                "vol": vol,
                "dd": round(dd, 4),
            },
        )
