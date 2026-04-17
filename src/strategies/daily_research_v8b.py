"""Keltner Channel Rejection — buy rejection candles at lower Keltner band.

Entry: Bar's low pierces below lower Keltner channel, but close is inside/above.
This signals a volatility-driven selloff that was rejected (buyers stepped in).
Requires low IBS < 0.35 (closed in lower part of range) for extra confirmation.
Skips HIGH/SHOCK vol, DOWN trend, earnings, FOMC.
Short hold, tight R:R — target is Keltner midline (EMA).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v8b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.atr_period = int(config.get("atr_period", 14))
        self.keltner_period = int(config.get("keltner_period", 20))
        self.keltner_mult = float(config.get("keltner_mult", 2.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.ibs_max = float(config.get("ibs_max", 0.35))
        self.sma_period = int(config.get("sma_period", 50))

    # --- Indicator helpers ---

    @staticmethod
    def _ema(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = (v - ema) * mult + ema
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
    def _ibs(bar: Bar) -> float:
        rng = bar.high - bar.low
        if rng < 1e-9:
            return 0.5
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

        # Skip HIGH and SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot and snapshot.vol in (VolRegime.SHOCK, VolRegime.HIGH):
            return None

        # Event filters
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        # Skip DOWN trend
        regime_trend = labels.get("regime_trend", "FLAT").upper()
        if regime_trend == "DOWN":
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Keltner channel
        keltner_ma = self._ema(closes, self.keltner_period)
        atr = self._atr(bars, self.atr_period)
        if keltner_ma is None or atr is None or atr < 1e-9:
            return None

        lower_keltner = keltner_ma - self.keltner_mult * atr

        # Core signal: bar's low went below lower Keltner, but close is above it
        # This is a "rejection candle" — sellers pushed below but buyers reclaimed
        if bar.low >= lower_keltner:
            return None  # Low didn't touch the band
        if bar.close < lower_keltner:
            return None  # Closed below = genuine breakdown, not rejection

        # IBS filter: close should be in lower portion of range (not too recovered)
        # This catches the "just barely recovered" candles which bounce next day
        ibs = self._ibs(bar)
        if ibs > self.ibs_max:
            return None

        # Trend safety: don't buy if too far below SMA(50) — deep decline
        sma = self._sma(closes, self.sma_period)
        if sma is not None and bar.close < sma * 0.92:
            return None

        # Target: Keltner midline (EMA), or a minimum of 1 ATR
        target = max(keltner_ma, bar.close + atr)
        stop = bar.close - self.stop_atr_mult * atr

        if target <= bar.close:
            return None

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop,
            target_price=target,
            meta={
                "regime_trend": regime_trend,
                "keltner_ma": round(keltner_ma, 2),
                "lower_keltner": round(lower_keltner, 2),
                "ibs": round(ibs, 3),
                "atr": round(atr, 4),
                "seed": "keltner_rejection",
            },
        )
