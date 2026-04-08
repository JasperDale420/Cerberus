"""Daily Research Strategy — momentum+RSI(2) with regime sizing, iter2."""

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
        self.tgt_m = float(config.get("target_atr_mult", 6.0))
        self.rsi_threshold = float(config.get("rsi_threshold", 32.0))
        self.breakout_period = int(config.get("breakout_period", 20))
        self.min_bars = int(config.get("min_bars", 55))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=100)
            self._h[s] = deque(maxlen=100)
            self._lo[s] = deque(maxlen=100)
            self._pd[s] = None
            self._dhlc[s] = [0.0, 0.0, 0.0]

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

        # Regime filter
        snap = ms.regime_snapshot
        trend = str(snap.trend.value).lower() if snap and snap.trend else ""
        vol = str(snap.vol.value).lower() if snap and snap.vol else ""

        # Skip DOWN trend, HIGH vol, SHOCK vol
        if trend == "down" or vol in ("high", "shock"):
            return None

        # Trend structure: price above SMA(20) and SMA(50)
        sma20 = sum(cl[-20:]) / 20
        if price <= sma20:
            return None

        if len(cl) >= 50:
            sma50 = sum(cl[-50:]) / 50
            if price <= sma50:
                return None
            # Confirm SMA(20) is rising (above its value 5 bars ago)
            if len(cl) >= 25:
                sma20_prev = sum(cl[-25:-5]) / 20
                if sma20 < sma20_prev:
                    return None

        # Positive 10-day momentum
        if len(cl) >= 11 and price <= cl[-11]:
            return None

        # RSI(2) — primary signal
        rsi2 = self._rsi(c, n=2)

        # Regime-adaptive thresholds and sizing
        if trend == "up":
            rsi_thr = self.rsi_threshold + 15.0
            size = 1.0
        elif trend == "flat":
            rsi_thr = self.rsi_threshold + 10.0
            size = 0.7
        else:
            rsi_thr = self.rsi_threshold
            size = 0.5

        # Scale down in normal vol (vs low vol)
        if vol == "normal":
            size *= 0.85

        rsi2_ok = rsi2 is not None and rsi2 < rsi_thr

        # RSI(14) secondary signal
        rsi14 = self._rsi(c, n=14)
        rsi14_ok = rsi14 is not None and rsi14 < rsi_thr

        # Breakout signal in uptrends
        bp = self.breakout_period
        prev = cl[-(bp + 1) : -1] if len(cl) > bp else []
        breakout_ok = trend == "up" and len(prev) >= bp and price > max(prev)

        if not rsi2_ok and not rsi14_ok and not breakout_ok:
            return None

        # Breakout gets slightly higher sizing
        if breakout_ok and not rsi2_ok:
            size *= 0.9

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
