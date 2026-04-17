"""Consecutive-down + IBS mean reversion with internal vol/trend filters.

Entry: 2+ consecutive down closes + low IBS + close below SMA(20).
Internal filters: ATR/price < 3% (skip volatile), ROC(20) > -10% (skip crashes).
Does NOT rely on regime labels (unreliable per-bar). Only SHOCK via market_state.
Symmetric ATR stop/target. Long-only, daily bars, max_hold_days=3.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v9a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.consec_down_min = int(config.get("consec_down_min", 2))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.35))
        self.sma_period = int(config.get("sma_period", 20))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        self.atr_pct_cap = float(config.get("atr_pct_cap", 0.03))
        self.roc_period = int(config.get("roc_period", 20))
        self.roc_floor = float(config.get("roc_floor", -0.05))
        self.sma_trend_period = int(config.get("sma_trend_period", 50))

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

    @staticmethod
    def _roc(closes: list[float], period: int) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        old = closes[-period - 1]
        if old < 1e-9:
            return None
        return (closes[-1] - old) / old

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

        # Skip earnings and FOMC
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # ATR for stop/target and vol filter
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Internal vol filter: ATR/price caps out high-vol periods
        atr_pct = atr / bar.close if bar.close > 0 else 0
        if atr_pct > self.atr_pct_cap:
            return None

        # Internal trend filter: skip if price crashed recently
        roc = self._roc(closes, self.roc_period)
        if roc is not None and roc < self.roc_floor:
            return None

        # Consecutive down days (hard gate)
        consec = self._consecutive_downs(closes)
        if consec < self.consec_down_min:
            return None

        # IBS — close near the low of the day
        ibs = self._ibs(bar)
        if ibs >= self.ibs_threshold:
            return None

        # Uptrend gate: close must be above SMA(50) — only buy dips in uptrends
        sma_trend = self._sma(closes, self.sma_trend_period)
        if sma_trend is not None and bar.close < sma_trend:
            return None

        # Close below SMA(20) — pullback entry
        sma = self._sma(closes, self.sma_period)
        if sma is None or bar.close >= sma:
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
                "atr_pct": round(atr_pct, 4),
                "roc": round(roc, 4) if roc else None,
                "seed": "mean_reversion",
            },
        )
