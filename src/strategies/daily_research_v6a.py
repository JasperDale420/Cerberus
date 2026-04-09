"""Daily Research v6a — 20-day high breakout with volume confirmation.

A trend-continuation strategy that naturally self-filters in bad markets:
- Buy when price breaks above 20-day high (momentum confirmation)
- Volume must be above average (institutional participation)
- SMA(50) uptrend filter
- Block SHOCK volatility
- ATR-based stops and targets
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
        self.target_atr_mult = float(config.get("target_atr_mult", 3.0))
        self.breakout_period = int(config.get("breakout_period", 20))
        self.vol_mult = float(config.get("vol_mult", 1.2))
        self.sma_period = int(config.get("sma_period", 50))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=250)
            self._h[s] = deque(maxlen=250)
            self._lo[s] = deque(maxlen=250)
            self._vol[s] = deque(maxlen=250)
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

        # SMA(50) uptrend filter
        sma = self._sma(cl, self.sma_period)
        if sma is None or price < sma:
            return None

        # 20-day high breakout: current close must exceed max of prior N closes
        bp = self.breakout_period
        if len(cl) < bp + 1:
            return None
        prior_high = max(cl[-(bp + 1) : -1])
        if price <= prior_high:
            return None

        # Volume confirmation: today's volume above average
        vl = list(self._vol[sym])
        if len(vl) < 20:
            return None
        avg_vol = sum(vl[-20:]) / 20
        if avg_vol <= 0 or vl[-1] < avg_vol * self.vol_mult:
            return None

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
