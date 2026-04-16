"""Iteration 8: IBS + Down Days with quality filters.

Buy after 2+ consecutive down closes when IBS is low.
Skip HIGH/SHOCK vol, Monday, near_earnings, near_fomc.
Volume confirmation. Tight drawdown (8%). Min dip magnitude (1% total decline).
ATR-based stops/targets.
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
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        self.min_down_days = int(config.get("min_down_days", 2))
        self.min_dip_pct = float(config.get("min_dip_pct", 0.01))
        self.vol_mult = float(config.get("vol_mult", 0.8))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.08))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.1))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
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
        volumes = [b.volume for b in bars]

        # --- Regime filter: skip HIGH and SHOCK vol ---
        labels = symbol_state.meta.get("regime_labels", {})
        vol = labels.get("regime_vol", "")
        if vol in ("HIGH", "SHOCK"):
            return None

        # --- Skip near earnings and FOMC ---
        if labels.get("near_earnings", False):
            return None
        if labels.get("near_fomc", False):
            return None

        # --- Day-of-week filter: skip Monday ---
        bar_time = bar.time
        if isinstance(bar_time, datetime):
            if bar_time.weekday() == 0:
                return None

        # --- IBS filter: close must be near the low of the day ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # --- Consecutive down days with min dip magnitude ---
        if len(closes) < self.min_down_days + 1:
            return None
        for i in range(1, self.min_down_days + 1):
            if closes[-i] >= closes[-i - 1]:
                return None
        # Total decline over the down days must be significant
        start_price = closes[-(self.min_down_days + 1)]
        if start_price > 0:
            total_decline = (start_price - closes[-1]) / start_price
            if total_decline < self.min_dip_pct:
                return None

        # --- Volume filter ---
        if len(volumes) >= 20:
            avg_vol = sum(volumes[-20:]) / 20
            if avg_vol > 0 and bar.volume < self.vol_mult * avg_vol:
                return None

        # --- Drawdown filter ---
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
                "total_decline": round(total_decline, 4),
                "atr": round(atr, 4),
                "vol_regime": vol,
                "dd": round(dd, 4),
            },
        )
