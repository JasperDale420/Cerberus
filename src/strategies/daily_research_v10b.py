"""Trend Pullback — Consec Down + IBS in Strong Uptrends.

Entry: SMA(10) > SMA(50) by min_spread + close > SMA(50) + 2+ consecutive
       down closes + IBS < threshold + adequate volume.
Filters: Block DOWN regime + SHOCK/HIGH vol, skip earnings/FOMC/quad_witch.
Risk: Stop 1.5 ATR, target optimized. Max hold 3 days.

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
        self.min_bars = int(config.get("min_bars", 55))
        self.sma_fast = int(config.get("sma_fast", 10))
        self.sma_slow = int(config.get("sma_slow", 50))
        self.consec_down_min = int(config.get("consec_down_min", 3))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        self.vol_avg_period = int(config.get("vol_avg_period", 20))
        self.vol_min_ratio = float(config.get("vol_min_ratio", 0.7))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        self.min_sma_spread = float(config.get("min_sma_spread", 0.01))

    # --- Indicator helpers ---

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

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
    def _consecutive_down_count(closes: list[float]) -> int:
        if len(closes) < 2:
            return 0
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

        # --- Regime filters ---
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol in (VolRegime.SHOCK, VolRegime.HIGH):
            return None

        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("trend_regime_symbol", "") == "DOWN":
            return None
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None
        if labels.get("quad_witch_week", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # --- Trend: SMA(10) > SMA(50) by min spread, close > SMA(50) ---
        sma_f = self._sma(closes, self.sma_fast)
        sma_s = self._sma(closes, self.sma_slow)
        if sma_f is None or sma_s is None:
            return None
        if sma_s < 1e-9:
            return None
        sma_spread = (sma_f - sma_s) / sma_s
        if sma_spread < self.min_sma_spread:
            return None
        if bar.close <= sma_s:
            return None

        # --- Pullback: consecutive down closes ---
        consec = self._consecutive_down_count(closes)
        if consec < self.consec_down_min:
            return None

        # --- IBS exhaustion ---
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # --- Volume confirmation ---
        avg_vol = self._sma(volumes, self.vol_avg_period)
        if avg_vol is None or avg_vol < 1e-9 or bar.volume < self.vol_min_ratio * avg_vol:
            return None

        # --- ATR for stop and target ---
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
                "consec_down": consec,
                "ibs": round(ibs, 3),
                "sma_fast": round(sma_f, 2),
                "sma_slow": round(sma_s, 2),
                "atr": round(atr, 4),
                "seed": "trend_pullback",
            },
        )
