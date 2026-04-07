"""Daily Research Strategy — momentum breakout, long-only."""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class DailyResearchStrategy(BaseStrategy):
    name = "daily_research_strategy"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._lo: dict[str, deque[float]] = {}
        self._pd: dict[str, date | None] = {}
        self._dhlc: dict[str, list[float]] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.stop_m = float(config.get("stop_atr_mult", 2.0))
        self.tgt_m = float(config.get("target_atr_mult", 4.0))
        self.sma_len = int(config.get("sma_len", 50))
        self.breakout_len = int(config.get("breakout_period", 10))
        self.min_bars = int(config.get("min_bars", 55))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s], self._h[s], self._lo[s] = deque(maxlen=80), deque(maxlen=80), deque(maxlen=80)
            self._pd[s] = None
            self._dhlc[s] = [0.0, 0.0, 0.0]

    def _atr(self, h: deque, lo: deque, c: deque, p: int = 14) -> float | None:
        if len(h) < p + 1:
            return None
        hl, ll, cl = list(h), list(lo), list(c)
        return sum(max(hl[i] - ll[i], abs(hl[i] - cl[i - 1]), abs(ll[i] - cl[i - 1])) for i in range(1, p + 1)) / p

    def on_bar(self, symbol: str, bar: Bar, symbol_state: SymbolState, market_state: MarketState) -> Optional[Signal]:
        self._init(symbol)
        dt = bar.time.date() if isinstance(bar.time, datetime) else bar.time
        d = self._dhlc[symbol]
        if self._pd[symbol] is not None and dt != self._pd[symbol]:
            self._c[symbol].append(d[2])
            self._h[symbol].append(d[0])
            self._lo[symbol].append(d[1])
            d[0], d[1], d[2] = bar.high, bar.low, bar.close
            sig = self._sig(symbol, bar, market_state)
            self._pd[symbol] = dt
            return sig
        if self._pd[symbol] is None:
            d[0], d[1] = bar.high, bar.low
        else:
            d[0] = max(d[0], bar.high)
            d[1] = min(d[1], bar.low)
        d[2] = bar.close
        self._pd[symbol] = dt
        return None

    def _sig(self, sym: str, bar: Bar, ms: MarketState) -> Signal | None:
        c, h, lo = self._c[sym], self._h[sym], self._lo[sym]
        if len(c) < self.min_bars or not self._check_cooldown(sym, bar.time):
            return None
        atr = self._atr(h, lo, c)
        if not atr or atr < 0.01:
            return None
        cl = list(c)
        price = cl[-1]
        # SMA trend filter: price must be above SMA
        sma = sum(cl[-self.sma_len :]) / self.sma_len
        if price <= sma:
            return None
        # Skip DOWN regimes
        snap = ms.regime_snapshot
        trend = str(snap.trend.value).lower() if snap and snap.trend else ""
        if trend == "down":
            return None
        # Momentum: new N-day high close
        prev_highs = cl[-(self.breakout_len + 1) : -1]
        if len(prev_highs) < self.breakout_len or price <= max(prev_highs):
            return None
        # Stop at recent swing low
        recent_lows = list(lo)[-self.breakout_len :]
        swing_low = min(recent_lows) if recent_lows else price - atr * self.stop_m
        stop = max(swing_low, price - atr * self.stop_m)  # floor at ATR stop
        risk = price - stop
        if risk < 0.01:
            return None
        target = price + risk * 2.0  # 2:1 R:R minimum
        self.last_signal_time[sym] = bar.time
        return Signal(
            symbol=sym,
            side=OrderSide.BUY,
            size_hint=0.0,
            entry_price=price,
            stop_price=stop,
            target_price=target,
            strategy=self.name,
            generated_at=bar.time,
        )
