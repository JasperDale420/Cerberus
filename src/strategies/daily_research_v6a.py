"""Daily Research v6a — Cumulative RSI(2) mean reversion.

Single-signal design for maximum consistency:
- Cumulative RSI(2) over 2 days (CumRSI) identifies extreme oversold
- SMA50 filter: only buy dips in structural uptrends
- ATR-based stops and targets with 2% hard cap on stops
- No regime gating beyond SHOCK vol block

Designed for walk-forward stability with minimal parameters.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv6aStrategy(BaseStrategy):
    name = "daily_research_v6a"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self._c: dict[str, deque[float]] = {}
        self._h: dict[str, deque[float]] = {}
        self._lo: dict[str, deque[float]] = {}
        self._vol: dict[str, deque[float]] = {}
        self._pd: dict[str, date | None] = {}
        self._dhlcv: dict[str, list[float]] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 5.0))
        self.rsi2_threshold = float(config.get("rsi2_threshold", 25))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=120)
            self._h[s] = deque(maxlen=120)
            self._lo[s] = deque(maxlen=120)
            self._vol[s] = deque(maxlen=120)
            self._pd[s] = None
            self._dhlcv[s] = [0.0, 0.0, 0.0, 0.0]

    def _atr(self, h: deque, lo: deque, c: deque, p: int = 14) -> float | None:
        if len(h) < p + 1:
            return None
        hl, ll, cl = list(h), list(lo), list(c)
        return (
            sum(max(hl[i] - ll[i], abs(hl[i] - cl[i - 1]), abs(ll[i] - cl[i - 1])) for i in range(len(hl) - p, len(hl)))
            / p
        )

    def _rsi(self, v: deque, n: int = 14) -> float | None:
        if len(v) < n + 1:
            return None
        d = list(v)
        g = sum(max(d[i] - d[i - 1], 0) for i in range(-n, 0))
        ls = sum(max(d[i - 1] - d[i], 0) for i in range(-n, 0))
        return 100.0 if ls == 0 else 100.0 - 100.0 / (1.0 + g / ls)

    def _sma(self, vals: list[float], period: int) -> float | None:
        if len(vals) < period:
            return None
        return sum(vals[-period:]) / period

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        self._init(symbol)
        dt = bar.time.date() if isinstance(bar.time, datetime) else bar.time
        d = self._dhlcv[symbol]
        if self._pd[symbol] is not None and dt != self._pd[symbol]:
            self._c[symbol].append(d[2])
            self._h[symbol].append(d[0])
            self._lo[symbol].append(d[1])
            self._vol[symbol].append(d[3])
            d[0], d[1], d[2], d[3] = bar.high, bar.low, bar.close, bar.volume
            sig = self._evaluate(symbol, bar, market_state)
            self._pd[symbol] = dt
            return sig
        if self._pd[symbol] is None:
            d[0], d[1] = bar.high, bar.low
        else:
            d[0] = max(d[0], bar.high)
            d[1] = min(d[1], bar.low)
        d[2] = bar.close
        d[3] = bar.volume
        self._pd[symbol] = dt
        return None

    def _evaluate(self, sym: str, bar: Bar, ms: MarketState) -> Signal | None:
        c = self._c[sym]
        if len(c) < self.min_bars or not self._check_cooldown(sym, bar.time):
            return None

        atr = self._atr(self._h[sym], self._lo[sym], c)
        if not atr or atr < 0.01:
            return None

        cl = list(c)
        price = cl[-1]

        # Block SHOCK volatility
        snap = ms.regime_snapshot
        vol = str(snap.vol.value).lower() if snap and snap.vol else ""
        if vol == "shock":
            return None

        # SMA50 gate: only buy dips in stocks with structural strength
        sma50 = self._sma(cl, 50)
        if sma50 is not None and price < sma50:
            return None

        # Cumulative RSI(2): sum of last 2 RSI(2) readings
        # More robust than single RSI(2) — filters out single-bar noise
        rsi2_now = self._rsi(c, n=2)
        if rsi2_now is None:
            return None

        # Compute RSI(2) for previous bar by using a shifted deque
        if len(c) < 4:
            return None
        prev_rsi_data = list(c)[:-1]
        g = sum(max(prev_rsi_data[i] - prev_rsi_data[i - 1], 0) for i in range(-2, 0))
        ls = sum(max(prev_rsi_data[i - 1] - prev_rsi_data[i], 0) for i in range(-2, 0))
        rsi2_prev = 100.0 if ls == 0 else 100.0 - 100.0 / (1.0 + g / ls)

        cum_rsi = rsi2_now + rsi2_prev

        # Cumulative RSI threshold: lower = more selective/consistent
        if cum_rsi < self.rsi2_threshold * 2:
            # Confirm: price must have dropped (bearish pressure)
            if price < cl[-2]:
                stop_dist = min(atr * self.stop_atr_mult, price * 0.02)
                stop = price - stop_dist
                target = price + atr * self.target_atr_mult

                self.last_signal_time[sym] = bar.time
                return Signal(
                    symbol=sym,
                    side=OrderSide.BUY,
                    size_hint=1.0,
                    entry_price=price,
                    stop_price=stop,
                    target_price=target,
                    strategy=self.name,
                    generated_at=bar.time,
                )

        return None
