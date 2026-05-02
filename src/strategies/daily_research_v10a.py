"""Pullback Mean Reversion — Best of 15 iterations (iter 10, min PF 0.61).

Entry: EMA(20) > EMA(50) uptrend + close pulled back below SMA(20)
       + 2+ consecutive down closes + IBS < 0.4 (seller exhaustion).
Filters: Block DOWN regime + SHOCK vol, skip earnings/FOMC/quad_witch.
Risk: Stop 1.5 ATR, target 3.0 ATR (2:1 R:R). Max hold 5 days.

Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedMeanReversionStrategy(BaseStrategy):
    name = "daily_research_v10a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.ema_fast = int(config.get("ema_fast", 20))
        self.ema_slow = int(config.get("ema_slow", 50))
        self.sma_pullback = int(config.get("sma_pullback", 20))
        self.consec_down_min = int(config.get("consec_down_min", 2))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.4))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicator helpers ---

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = v * mult + ema * (1 - mult)
        return ema

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

        # Skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        # Block DOWN regime (market-level)
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("trend_regime_symbol", "") == "DOWN":
            return None

        # Skip earnings, FOMC, quad witch
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None
        if labels.get("quad_witch_week", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Trend: EMA(20) > EMA(50) — uptrend
        ema_f = self._ema(closes, self.ema_fast)
        ema_s = self._ema(closes, self.ema_slow)
        if ema_f is None or ema_s is None:
            return None
        if ema_f <= ema_s:
            return None

        # Pullback: close below SMA(20) — dipped below short-term mean
        sma_pb = self._sma(closes, self.sma_pullback)
        if sma_pb is None or bar.close >= sma_pb:
            return None

        # Consecutive down days
        consec = self._consecutive_down_count(closes)
        if consec < self.consec_down_min:
            return None

        # IBS: close near day's low
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # ATR for stop and target
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
                "ema_fast": round(ema_f, 2),
                "ema_slow": round(ema_s, 2),
                "atr": round(atr, 4),
                "seed": "mean_reversion",
            },
        )
