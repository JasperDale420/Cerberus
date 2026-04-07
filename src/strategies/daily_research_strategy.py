"""Daily Research Strategy — buy dips in strong uptrends with momentum confirmation."""

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
        self.stop_m = float(config.get("stop_atr_mult", 1.5))
        self.tgt_m = float(config.get("target_atr_mult", 2.5))
        self.rsi_threshold = float(config.get("rsi_threshold", 45.0))
        self.breakout_period = int(config.get("breakout_period", 20))
        self.min_bars = int(config.get("min_bars", 25))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=80)
            self._h[s] = deque(maxlen=80)
            self._lo[s] = deque(maxlen=80)
            self._pd[s] = None
            self._dhlc[s] = [0.0, 0.0, 0.0]

    def _atr(self, h: deque, lo: deque, c: deque, p: int = 14) -> float | None:
        if len(h) < p + 1:
            return None
        hl, ll, cl = list(h), list(lo), list(c)
        return sum(max(hl[i] - ll[i], abs(hl[i] - cl[i - 1]), abs(ll[i] - cl[i - 1])) for i in range(1, p + 1)) / p

    def _rsi(self, v: deque, n: int = 14) -> float | None:
        if len(v) < n + 1:
            return None
        d = list(v)
        g = sum(max(d[i] - d[i - 1], 0) for i in range(-n, 0))
        ls = sum(max(d[i - 1] - d[i], 0) for i in range(-n, 0))
        return 100.0 if ls == 0 else 100.0 - 100.0 / (1.0 + g / ls)

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
        c = self._c[sym]
        if len(c) < self.min_bars or not self._check_cooldown(sym, bar.time):
            return None
        atr = self._atr(self._h[sym], self._lo[sym], c)
        if not atr or atr < 0.01:
            return None

        cl = list(c)
        price = cl[-1]

        # Regime gate: skip DOWN and SHOCK
        snap = ms.regime_snapshot
        trend = str(snap.trend.value).lower() if snap and snap.trend else ""
        vol = str(snap.vol.value).lower() if snap and snap.vol else ""
        if trend == "down" or vol == "shock":
            return None

        # Price must be above SMA(20) — uptrend confirmation
        sma20 = sum(cl[-20:]) / 20
        if price <= sma20:
            return None

        # 10-day positive momentum — price higher than 10 days ago
        if len(cl) >= 11 and price <= cl[-11]:
            return None

        # Signal 1: RSI(2) extreme oversold — Connors-style snap-back
        rsi2 = self._rsi(c, n=2)
        rsi2_ok = rsi2 is not None and rsi2 < self.rsi_threshold

        # Signal 2: RSI(14) moderate pullback in uptrend
        rsi14 = self._rsi(c, n=14)
        rsi14_ok = rsi14 is not None and rsi14 < self.rsi_threshold

        # Signal 3: Price dip below SMA(5) while above SMA(20) — buy the dip
        sma5 = sum(cl[-5:]) / 5
        dip_ok = price < sma5 and price > sma20

        # Signal 4: Breakout — new N-day high close (trend=up only)
        bp = self.breakout_period
        prev = cl[-(bp + 1) : -1] if len(cl) > bp else []
        breakout_ok = trend == "up" and len(prev) >= bp and price > max(prev)

        if not rsi2_ok and not rsi14_ok and not dip_ok and not breakout_ok:
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
