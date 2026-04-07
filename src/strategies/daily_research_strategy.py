"""Daily Research Strategy — RSI(2) dip-buy with trend + bounce confirmation.

Long-only. Buy RSI(2) oversold dips in uptrends with bounce confirmation.
Skip DOWN and SHOCK regimes. Wider targets for compounding gains.
"""

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
        self.sma_slow = int(config.get("sma_slow", 50))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_buy = float(config.get("rsi_buy_threshold", 15.0))
        self.stop_m = float(config.get("stop_atr_mult", 1.5))
        self.tgt_m = float(config.get("target_atr_mult", 4.0))
        self.min_bars = int(config.get("min_bars", 55))
        self.atr_period = int(config.get("atr_period", 14))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            mx = max(self.sma_slow + 10, 80)
            self._c[s] = deque(maxlen=mx)
            self._h[s] = deque(maxlen=mx)
            self._lo[s] = deque(maxlen=mx)
            self._pd[s] = None
            self._dhlc[s] = [0.0, 0.0, 0.0]

    def _atr(self, h: deque, lo: deque, c: deque) -> float | None:
        p = self.atr_period
        if len(h) < p + 1:
            return None
        hl, ll, cl = list(h), list(lo), list(c)
        return (
            sum(max(hl[i] - ll[i], abs(hl[i] - cl[i - 1]), abs(ll[i] - cl[i - 1])) for i in range(len(hl) - p, len(hl)))
            / p
        )

    def _rsi(self, v: deque, n: int) -> float | None:
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

        rsi2 = self._rsi(c, self.rsi_period)
        if rsi2 is None:
            return None

        cl = list(c)
        price = cl[-1]

        # Trend filter: price above slow SMA
        sma_s = sum(cl[-self.sma_slow :]) / self.sma_slow
        if price <= sma_s:
            return None

        # Regime gating
        snap = ms.regime_snapshot
        trend = str(snap.trend.value).lower() if snap and snap.trend else ""
        vol = str(snap.vol.value).lower() if snap and snap.vol else ""
        if vol == "shock" or trend == "down":
            return None

        # RSI(2) oversold dip-buy with bounce confirmation
        if rsi2 >= self.rsi_buy:
            return None

        # Bounce confirmation: today's close > yesterday's close
        if len(cl) >= 2 and price <= cl[-2]:
            return None

        stop = price - atr * self.stop_m
        target = price + atr * self.tgt_m
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
