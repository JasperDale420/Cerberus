"""Strong Oversold Bounce: 3+ Consecutive Down + Low IBS + Trend Health.

Buy after 3+ consecutive down closes when IBS < 0.2 (extreme selling exhaustion)
and price is above SMA(50) (not in structural downtrend).
Uses asymmetric R:R (1.5 ATR stop, 3.0 ATR target).
Skips SHOCK vol and earnings. Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v9b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.consec_down_min = int(config.get("consec_down_min", 3))
        self.ibs_max = float(config.get("ibs_max", 0.20))
        self.sma_long_period = int(config.get("sma_long_period", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))

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
    def _consecutive_down(closes: list[float]) -> int:
        """Count consecutive down closes from the end."""
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i - 1]:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _ibs(bar: Bar) -> Optional[float]:
        """Internal Bar Strength: (close - low) / (high - low)."""
        rng = bar.high - bar.low
        if rng < 1e-9:
            return None
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

        # Regime label filtering
        labels = symbol_state.meta.get("regime_labels", {})
        vol_regime = str(labels.get("regime_vol", "NORMAL")).upper()
        if vol_regime == "SHOCK":
            return None

        # Skip near earnings
        if labels.get("near_earnings", False):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # Condition 1: 3+ consecutive down closes (strong oversold signal)
        consec = self._consecutive_down(closes)
        if consec < self.consec_down_min:
            return None

        # Condition 2: IBS < 0.20 (extreme selling exhaustion — closed at the low)
        ibs = self._ibs(bar)
        if ibs is None or ibs > self.ibs_max:
            return None

        # Condition 3: Price above SMA(50) (not in structural downtrend)
        sma_long = self._sma(closes, self.sma_long_period)
        if sma_long is not None and bar.close < sma_long:
            return None

        # ATR for stops and targets
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
                "mode": "strong_oversold_bounce",
                "consec_down": consec,
                "ibs": round(ibs, 3),
                "atr": round(atr, 4),
            },
        )
