"""Multi-Day Selloff Recovery — buy after N consecutive down days.

Core logic: stocks that close lower for 2+ days with low IBS (near day's low)
tend to bounce. Filter with SMA trend support and ATR vol gate.
Single unified entry across all regimes, no mode switching.
Tighter filters in high realized vol. Long-only, daily bars.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, VolRegime
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class SeedTrendPullbackStrategy(BaseStrategy):
    name = "daily_research_v7b"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        # Consecutive down days
        self.min_down_days = int(config.get("min_down_days", 2))
        # IBS threshold
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        # Trend support — price must be above SMA(N)
        self.sma_period = int(config.get("sma_period", 40))
        # SMA slope lookback — SMA must not be falling
        self.sma_slope_lookback = int(config.get("sma_slope_lookback", 5))
        # ATR params
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        # Vol filters
        self.max_atr_pct = float(config.get("max_atr_pct", 0.03))
        self.max_realized_vol = float(config.get("max_realized_vol", 0.25))
        # Max stop as pct of price
        self.max_stop_pct = float(config.get("max_stop_pct", 0.03))

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

    @staticmethod
    def _consecutive_down(closes: list[float], n: int) -> bool:
        if len(closes) < n + 1:
            return False
        for i in range(1, n + 1):
            if closes[-i] >= closes[-i - 1]:
                return False
        return True

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

        # Realized vol hard gate
        if market_state.realized_vol and market_state.realized_vol > self.max_realized_vol:
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]

        # ATR-based vol filter
        atr = self._atr(bars, self.atr_period)
        if atr is None or atr < 1e-9:
            return None
        if atr / bar.close > self.max_atr_pct:
            return None

        # Core signal: consecutive down days
        if not self._consecutive_down(closes, self.min_down_days):
            return None

        # IBS — close near the low (potential exhaustion)
        ibs = self._ibs(bar)
        if ibs > self.ibs_threshold:
            return None

        # Trend support — price above SMA
        sma = self._sma(closes, self.sma_period)
        if sma is not None and bar.close < sma:
            return None

        # SMA slope filter — SMA must be flat or rising (not in freefall)
        if sma is not None and len(closes) >= self.sma_period + self.sma_slope_lookback:
            sma_prev = self._sma(closes[: -self.sma_slope_lookback], self.sma_period)
            if sma_prev is not None and sma < sma_prev * 0.99:
                return None

        # Stop and target with cap
        raw_stop_dist = self.stop_atr_mult * atr
        max_stop_dist = bar.close * self.max_stop_pct
        stop_dist = min(raw_stop_dist, max_stop_dist)
        stop = bar.close - stop_dist
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
                "mode": "selloff_recovery",
                "down_days": self.min_down_days,
                "ibs": round(ibs, 2),
            },
        )
