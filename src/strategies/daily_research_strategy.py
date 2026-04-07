"""Daily Research Strategy — momentum + mean-reversion hybrid with regime sizing."""

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
        self._v: dict[str, deque[float]] = {}
        self._pd: dict[str, date | None] = {}
        self._dhlcv: dict[str, list[float]] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.stop_m = float(config.get("stop_atr_mult", 1.25))
        self.tgt_m = float(config.get("target_atr_mult", 10.0))
        self.rsi_ob = float(config.get("rsi_oversold", 25.0))
        self.rsi_ob_up = float(config.get("rsi_oversold_uptrend", 40.0))
        self.mom_period = int(config.get("momentum_period", 20))
        self.sma_fast = int(config.get("sma_fast", 10))
        self.sma_slow = int(config.get("sma_slow", 50))
        self.min_bars = int(config.get("min_bars", 55))
        self.vol_mult = float(config.get("volume_surge_mult", 1.0))
        self.breakout_period = int(config.get("breakout_period", 20))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=100)
            self._h[s] = deque(maxlen=100)
            self._lo[s] = deque(maxlen=100)
            self._v[s] = deque(maxlen=100)
            self._pd[s] = None
            self._dhlcv[s] = [0.0, 0.0, 0.0, 0.0]

    def _atr(self, h: deque, lo: deque, c: deque, p: int = 14) -> float | None:
        if len(h) < p + 1:
            return None
        hl, ll, cl = list(h), list(lo), list(c)
        return sum(max(hl[i] - ll[i], abs(hl[i] - cl[i - 1]), abs(ll[i] - cl[i - 1])) for i in range(1, p + 1)) / p

    def _rsi(self, v: deque, n: int = 2) -> float | None:
        if len(v) < n + 1:
            return None
        d = list(v)
        g = sum(max(d[i] - d[i - 1], 0) for i in range(-n, 0))
        ls = sum(max(d[i - 1] - d[i], 0) for i in range(-n, 0))
        return 100.0 if ls == 0 else 100.0 - 100.0 / (1.0 + g / ls)

    def _sma(self, cl: list[float], n: int) -> float | None:
        if len(cl) < n:
            return None
        return sum(cl[-n:]) / n

    def on_bar(self, symbol: str, bar: Bar, symbol_state: SymbolState, market_state: MarketState) -> Optional[Signal]:
        self._init(symbol)
        dt = bar.time.date() if isinstance(bar.time, datetime) else bar.time
        d = self._dhlcv[symbol]
        if self._pd[symbol] is not None and dt != self._pd[symbol]:
            self._c[symbol].append(d[2])
            self._h[symbol].append(d[0])
            self._lo[symbol].append(d[1])
            self._v[symbol].append(d[3])
            d[0], d[1], d[2], d[3] = bar.high, bar.low, bar.close, bar.volume
            sig = self._sig(symbol, bar, market_state)
            self._pd[symbol] = dt
            return sig
        if self._pd[symbol] is None:
            d[0], d[1] = bar.high, bar.low
        else:
            d[0] = max(d[0], bar.high)
            d[1] = min(d[1], bar.low)
        d[2] = bar.close
        d[3] = d[3] + bar.volume
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

        # Regime filter: skip DOWN trend and SHOCK vol
        snap = ms.regime_snapshot
        trend = str(snap.trend.value).lower() if snap and snap.trend else ""
        vol = str(snap.vol.value).lower() if snap and snap.vol else ""
        if vol == "shock":
            return None

        # Trend structure: price above slow SMA
        sma_s = self._sma(cl, self.sma_slow)
        if sma_s is None:
            return None
        if price <= sma_s:
            return None

        # Fast SMA for trend direction
        sma_f = self._sma(cl, self.sma_fast)
        if sma_f is None:
            return None

        # Momentum: positive N-day return
        if len(cl) > self.mom_period:
            mom = (price - cl[-self.mom_period - 1]) / cl[-self.mom_period - 1]
        else:
            mom = 0.0

        # RSI(2) — primary mean-reversion signal
        rsi2 = self._rsi(c, n=2)
        if rsi2 is None:
            return None

        # Signal logic depends on regime
        signal_ok = False
        size = 0.0

        if trend == "up":
            # In uptrend: buy RSI(2) dips or breakouts — aggressive
            if rsi2 < self.rsi_ob_up and mom > 0.0:
                signal_ok = True
                size = 1.0  # max conviction
            elif self._is_breakout(cl, price):
                signal_ok = True
                size = 0.9
        elif trend == "flat":
            # In flat: only deep RSI(2) oversold with positive momentum
            if rsi2 < self.rsi_ob and mom > 0.02 and sma_f > sma_s:
                signal_ok = True
                size = 0.7
        elif trend == "down":
            # In downtrend: only extreme oversold bounces
            if rsi2 < 10.0 and mom > 0.05 and price > sma_f:
                signal_ok = True
                size = 0.4

        if not signal_ok:
            return None

        # Scale down in high vol
        if vol == "high":
            size *= 0.6

        self.last_signal_time[sym] = bar.time
        return Signal(
            symbol=sym,
            side=OrderSide.BUY,
            size_hint=size,
            entry_price=price,
            stop_price=price - atr * self.stop_m,
            target_price=price + atr * self.tgt_m,
            strategy=self.name,
            generated_at=bar.time,
        )

    def _is_breakout(self, cl: list[float], price: float) -> bool:
        bp = self.breakout_period
        if len(cl) <= bp:
            return False
        prev = cl[-(bp + 1) : -1]
        return price > max(prev)
