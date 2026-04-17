"""Volatility Fade — Buy after high-vol selloff days.

Evolved from seed_vol_breakout (volatility family). Instead of buying breakouts,
we fade volatility spikes: when ATR expands AND price drops significantly,
capitulation selling creates bounce opportunities.

Entry:
1. ATR expanding (vol spike) — current ATR > atr_expansion_mult * avg ATR
2. Price dropped: close < close[lookback] AND close near daily low
3. Close-to-low ratio (IBS-like): (close - low) / (high - low) < ibs_threshold
4. Trend not fighting: regime not DOWN or price above longer SMA
5. Skip earnings, FOMC, SHOCK

Exit: ATR-based stop/target, max hold days.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v9c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 30))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_avg_period = int(config.get("atr_avg_period", 20))
        self.atr_expansion_mult = float(config.get("atr_expansion_mult", 1.2))
        self.drop_lookback = int(config.get("drop_lookback", 3))
        self.min_drop_pct = float(config.get("min_drop_pct", 0.02))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        self.sma_period = int(config.get("sma_period", 50))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
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

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    def _atr_series(self, bars: list[Bar], period: int, count: int) -> list[float]:
        result = []
        for i in range(count):
            end_idx = len(bars) - count + i + 1
            if end_idx < period + 1:
                continue
            sub = bars[:end_idx]
            val = self._atr(sub, period)
            if val is not None:
                result.append(val)
        return result

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

        regime_labels = symbol_state.meta.get("regime_labels", {})
        if regime_labels.get("near_earnings"):
            return None
        if regime_labels.get("near_fomc"):
            return None

        regime_vol = regime_labels.get("regime_vol", "NORMAL")
        if regime_vol == "SHOCK":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        if len(closes) < max(self.sma_period, self.atr_avg_period + self.atr_period) + 1:
            return None

        # ATR and expansion check
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        atr_values = self._atr_series(bars, self.atr_period, self.atr_avg_period)
        if len(atr_values) < self.atr_avg_period:
            return None
        atr_avg = sum(atr_values) / len(atr_values)
        if atr_avg < 1e-9 or atr < self.atr_expansion_mult * atr_avg:
            return None

        # Price has dropped over lookback period
        if len(closes) < self.drop_lookback + 1:
            return None
        ref_close = closes[-(self.drop_lookback + 1)]
        if ref_close < 1e-9:
            return None
        drop_pct = (ref_close - bar.close) / ref_close
        if drop_pct < self.min_drop_pct:
            return None

        # IBS-like: close near daily low (capitulation)
        daily_range = bar.high - bar.low
        if daily_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / daily_range
        if ibs > self.ibs_threshold:
            return None

        # Trend filter: price above long-term SMA (not fighting major downtrend)
        sma = self._sma(closes, self.sma_period)
        if sma is not None and bar.close < sma * 0.95:
            return None  # Too far below SMA — real downtrend, not a dip

        # Stop and target
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
                "atr_expansion": round(atr / atr_avg, 2),
                "drop_pct": round(drop_pct * 100, 2),
                "ibs": round(ibs, 3),
                "regime_vol": regime_vol,
                "seed": "vol_fade_v1",
            },
        )
