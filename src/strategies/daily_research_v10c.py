"""NR4 Volatility Contraction Breakout — evolved from vol_breakout seed.

Buy after a Narrow Range 4 (NR4) day when the next day confirms upward
breakout (close > NR4 bar's high). Volatility contraction precedes
expansion — this is the vol_breakout DNA from the setup side.

Uses ATR for stop/target sizing. Adds trend context (close > SMA50).
Filters: Skip SHOCK, skip earnings/FOMC.
Long-only, daily bars, max_hold_days=5.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedVolBreakoutStrategy(BaseStrategy):
    name = "daily_research_v10c"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.atr_period = int(config.get("atr_period", 14))
        self.nr_lookback = int(config.get("nr_lookback", 4))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.trend_sma_period = int(config.get("trend_sma_period", 50))

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

    @staticmethod
    def _sma(values: list[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

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
        if snapshot and snapshot.vol == VolRegime.SHOCK:
            return None

        labels = symbol_state.meta.get("regime_labels", {})
        if labels.get("near_earnings", False):
            return None
        if labels.get("near_fomc", False):
            return None

        bars = list(symbol_state.bars)
        if len(bars) < self.nr_lookback + 2:
            return None

        closes = [b.close for b in bars]

        # Trend filter: close above SMA(50) for uptrend context
        sma_trend = self._sma(closes, self.trend_sma_period)
        if sma_trend is None or bar.close <= sma_trend:
            return None

        # Check if PREVIOUS bar was NR4 (narrowest range of last nr_lookback days)
        prev_bar = bars[-2]
        prev_range = prev_bar.high - prev_bar.low
        if prev_range < 1e-9:
            return None

        # Compare previous bar's range to the nr_lookback-1 bars before it
        is_nr = True
        for i in range(3, self.nr_lookback + 2):
            if len(bars) < i + 1:
                is_nr = False
                break
            comp_bar = bars[-i]
            comp_range = comp_bar.high - comp_bar.low
            if prev_range >= comp_range:
                is_nr = False
                break

        if not is_nr:
            return None

        # Breakout confirmation: today's close > NR bar's high
        if bar.close <= prev_bar.high:
            return None

        # ATR for stop/target
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
                "atr": round(atr, 4),
                "nr_range": round(prev_range, 4),
                "nr_high": round(prev_bar.high, 2),
                "seed": "vol_breakout",
            },
        )
