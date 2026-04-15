"""Cumulative Return Mean Reversion — buy multi-day drops.

Core signal: stock has dropped > drop_pct over lookback_days.
This captures multi-day selloffs that tend to revert, not single-bar anomalies.

Different from IBS (single bar) and RSI (momentum oscillator).
Uses cumulative return which is a simple, stable signal.

Add max_stop_pct to cap per-trade risk.
Skip SHOCK vol, earnings, FOMC.
Exclude leveraged/inverse ETFs (VXX, SQQQ, etc.) that don't mean-revert.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy

# Leveraged and inverse ETFs that don't mean-revert normally
_EXCLUDED = {"VXX", "UVXY", "SQQQ", "TQQQ", "SPXU", "SPXS", "SDOW", "LABU", "LABD"}


class dailyresearchv7dStrategy(BaseStrategy):
    name = "daily_research_v7d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 30))
        self.lookback_days = int(config.get("lookback_days", 5))
        self.drop_pct = float(config.get("drop_pct", 0.05))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.max_stop_pct = float(config.get("max_stop_pct", 0.025))

    # --- Indicator helpers ---

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
        # Exclude leveraged/inverse ETFs
        if symbol in _EXCLUDED:
            return None

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

        if len(closes) < max(self.min_bars, self.lookback_days + 1):
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

        # Core signal: cumulative return over lookback_days
        past_close = closes[-(self.lookback_days + 1)]
        cum_return = (bar.close / past_close) - 1.0

        # Buy only if stock has dropped at least drop_pct
        if cum_return >= -self.drop_pct:
            return None

        # Stop and target
        stop_atr = bar.close - self.stop_atr_mult * atr
        max_stop = bar.close * (1.0 - self.max_stop_pct)
        stop = max(stop_atr, max_stop)

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
                "cum_return": round(cum_return, 4),
                "lookback": self.lookback_days,
                "atr": round(atr, 4),
                "seed": "cumreturn_mean_reversion",
            },
        )
