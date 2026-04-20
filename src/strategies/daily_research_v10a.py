"""Seed: Multi-Factor Mean Reversion.

RSI(2) + Bollinger Band + IBS (Internal Bar Strength) with drawdown filter.
Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
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
        self.min_bars = int(config.get("min_bars", 50))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_threshold = float(config.get("rsi_threshold", 25.0))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.5))
        self.drawdown_lookback = int(config.get("drawdown_lookback", 40))
        self.drawdown_max = float(config.get("drawdown_max", 0.12))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))

    # --- Indicator helpers ---

    @staticmethod
    def _rsi(closes: list[float], period: int) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        gains = []
        losses = []
        for i in range(-period, 0):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss < 1e-9:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    @staticmethod
    def _std(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        subset = values[-period:]
        mean = sum(subset) / period
        variance = sum((v - mean) ** 2 for v in subset) / period
        return variance ** 0.5

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

        # RSI(2) filter
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_threshold:
            return None

        # Bollinger Band: compute for target (midline) — not a hard gate
        sma = self._sma(closes, self.bb_period)
        std = self._std(closes, self.bb_period)
        if sma is None or std is None or std < 1e-9:
            return None

        # IBS (Internal Bar Strength): must be low
        bar_range = bar.high - bar.low
        if bar_range < 1e-9:
            return None
        ibs = (bar.close - bar.low) / bar_range
        if ibs >= self.ibs_threshold:
            return None

        # Drawdown filter: skip if price dropped > drawdown_max from lookback high
        lookback_highs = [b.high for b in bars[-self.drawdown_lookback :]]
        peak = max(lookback_highs)
        if peak > 0 and (peak - bar.close) / peak > self.drawdown_max:
            return None

        # ATR for stop
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None

        stop = bar.close - self.stop_atr_mult * atr
        target = sma  # BB midline (SMA20)

        # Only enter if target is above entry (positive expectancy)
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
                "rsi2": round(rsi, 2),
                "ibs": round(ibs, 3),
                "lower_bb": round(lower_bb, 2),
                "atr": round(atr, 4),
                "drawdown_from_peak": round((peak - bar.close) / peak, 4),
                "seed": "mean_reversion",
            },
        )
