"""IBS Mean Reversion with Trend Filter.

Core signal: Internal Bar Strength (IBS) < threshold AND price above SMA(50).
The SMA filter ensures we only buy oversold bounces in uptrending stocks,
avoiding falling knives in bear markets.

IBS near 0 = closed near day's low = oversold bounce expected.
Require 2+ consecutive down closes for confirmation.

Regime adaptation:
  UP   — wider target (1.5x scale)
  FLAT — standard
  DOWN — skip entirely (long-only can't profit in sustained downtrends)

Long-only. Skip SHOCK vol, earnings, FOMC.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv7dStrategy(BaseStrategy):
    name = "daily_research_v7d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.2))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.sma_period = int(config.get("sma_period", 50))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.03))

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
    def _ibs(bar: Bar) -> Optional[float]:
        """Internal Bar Strength: (close - low) / (high - low)."""
        rng = bar.high - bar.low
        if rng < 1e-9:
            return None
        return (bar.close - bar.low) / rng

    def _count_consecutive_down(self, closes: list[float]) -> int:
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

        # Skip earnings and FOMC
        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False) or labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        if len(closes) < self.min_bars:
            return None

        # ATR
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        # Min price filter
        if bar.close < 5.0:
            return None

        # ATR/price filter: skip dead stocks
        if atr / bar.close < 0.005:
            return None

        # Trend filter: price must be above SMA(50)
        # This is the KEY filter — it prevents buying falling knives
        sma = self._sma(closes, self.sma_period)
        if sma is None or bar.close <= sma:
            return None

        # IBS signal (closed near day's low)
        ibs = self._ibs(bar)
        if ibs is None or ibs >= self.ibs_threshold:
            return None

        # Consecutive down days confirmation (fixed at 2)
        consec = self._count_consecutive_down(closes)
        if consec < 2:
            return None

        # Regime-adaptive target
        regime_trend = labels.get("regime_trend", "FLAT").upper()
        target_mult = self.target_atr_mult
        stop_mult = self.stop_atr_mult

        if regime_trend == "UP":
            target_mult *= 1.5  # Let winners run in uptrends

        # ATR-based stop with max_stop_pct cap
        stop_atr = bar.close - stop_mult * atr
        max_stop = bar.close * (1.0 - self.max_stop_pct)
        stop = max(stop_atr, max_stop)

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
                "consec_down": consec,
                "regime": regime_trend,
                "above_sma50": True,
                "atr": round(atr, 4),
                "seed": "ibs_sma_filter",
            },
        )
