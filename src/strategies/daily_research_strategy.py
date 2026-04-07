"""Daily Research Strategy — dual signal: RSI reversion + close breakout."""

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
        self._dh: dict[str, float] = {}
        self._dl: dict[str, float] = {}
        self._dc: dict[str, float] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.atr_p = int(config.get("atr_period", 14))
        self.stop_m = float(config.get("stop_atr_mult", 2.0))
        self.tgt_m = float(config.get("target_atr_mult", 3.0))
        self.rsi_t = float(config.get("rsi_threshold", 40.0))
        self.bp = int(config.get("breakout_period", 20))
        self.min_bars = int(config.get("min_bars", 25))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.allow_overnight = True

    def _is(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=60)
            self._h[s] = deque(maxlen=60)
            self._lo[s] = deque(maxlen=60)
            self._pd[s] = None
            self._dh[s] = self._dl[s] = self._dc[s] = 0.0

    def _atr(self, h: deque, lo: deque, c: deque) -> float | None:
        if len(h) < self.atr_p + 1:
            return None
        hl, ll, cl = list(h), list(lo), list(c)
        p = self.atr_p
        return sum(max(hl[i] - ll[i], abs(hl[i] - cl[i - 1]), abs(ll[i] - cl[i - 1])) for i in range(1, p + 1)) / p

    def _rsi(self, v: deque) -> float | None:
        if len(v) < 15:
            return None
        d = list(v)
        g = sum(max(d[i] - d[i - 1], 0) for i in range(-14, 0))
        ls = sum(max(d[i - 1] - d[i], 0) for i in range(-14, 0))
        return 100.0 if ls == 0 else 100.0 - 100.0 / (1.0 + g / ls)

    def on_bar(self, symbol: str, bar: Bar, symbol_state: SymbolState, market_state: MarketState) -> Optional[Signal]:
        self._is(symbol)
        dt = bar.time.date() if isinstance(bar.time, datetime) else bar.time
        if self._pd[symbol] is not None and dt != self._pd[symbol]:
            self._c[symbol].append(self._dc[symbol])
            self._h[symbol].append(self._dh[symbol])
            self._lo[symbol].append(self._dl[symbol])
            self._dh[symbol], self._dl[symbol], self._dc[symbol] = bar.high, bar.low, bar.close
            sig = self._sig(symbol, bar, market_state)
            self._pd[symbol] = dt
            return sig
        if self._pd[symbol] is None:
            self._dh[symbol], self._dl[symbol] = bar.high, bar.low
        else:
            self._dh[symbol] = max(self._dh[symbol], bar.high)
            self._dl[symbol] = min(self._dl[symbol], bar.low)
        self._dc[symbol] = bar.close
        self._pd[symbol] = dt
        return None

    def _sig(self, sym: str, bar: Bar, ms: MarketState) -> Signal | None:
        c, h, lo = self._c[sym], self._h[sym], self._lo[sym]
        if len(c) < self.min_bars or not self._check_cooldown(sym, bar.time):
            return None
        snap = ms.regime_snapshot
        if snap and snap.trend and str(snap.trend.value).lower() == "down":
            return None
        cl = list(c)
        s20 = sum(cl[-20:]) / 20 if len(cl) >= 20 else None
        s20p = sum(cl[-25:-5]) / 20 if len(cl) >= 25 else None
        if s20 is not None and s20p is not None and s20 < s20p:
            return None
        atr = self._atr(h, lo, c)
        if atr is None or atr < 0.01:
            return None
        price = cl[-1]
        rsi = self._rsi(c)
        r_ok = rsi is not None and rsi < self.rsi_t
        pc = cl[-(self.bp + 1) : -1] if len(cl) > self.bp else []
        b_ok = len(pc) >= self.bp and price > max(pc)
        if not r_ok and not b_ok:
            return None
        self.last_signal_time[sym] = bar.time
        return Signal(
            symbol=sym,
            side=OrderSide.BUY,
            size_hint=0.0,
            entry_price=price,
            stop_price=price - atr * self.stop_m,
            target_price=price + atr * self.tgt_m,
            strategy=self.name,
            generated_at=bar.time,
        )
