"""Daily Research v6a — RSI(2) + IBS + down days, relaxed threshold.

Triple confirmation mean reversion:
- 2+ consecutive lower closes (exhaustion selling)
- RSI(2) < 15 oversold (relaxed for more trades)
- IBS < 0.3 (closed in bottom 30% of daily range)
- 1.5 ATR / 2% hard stop, max 5-day hold
- Only blocks SHOCK volatility
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
        self._pd: dict[str, date | None] = {}
        self._dhlcv: dict[str, list[float]] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 20))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.5))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.5))
        self.rsi2_threshold = float(config.get("rsi2_threshold", 15))
        self.ibs_threshold = float(config.get("ibs_threshold", 0.3))
        self.down_days = int(config.get("down_days", 2))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=60)
            self._h[s] = deque(maxlen=60)
            self._lo[s] = deque(maxlen=60)
            self._pd[s] = None
            self._dhlcv[s] = [0.0, 0.0, 0.0]

    def _atr(self, h: deque, lo: deque, c: deque, p: int = 14) -> float | None:
        if len(h) < p + 1:
            return None
        hl, ll, cl = list(h), list(lo), list(c)
        return (
            sum(max(hl[i] - ll[i], abs(hl[i] - cl[i - 1]), abs(ll[i] - cl[i - 1])) for i in range(len(hl) - p, len(hl)))
            / p
        )

    def _rsi(self, v: deque, n: int = 2) -> float | None:
        if len(v) < n + 1:
            return None
        d = list(v)
        g = sum(max(d[i] - d[i - 1], 0) for i in range(-n, 0))
        ls = sum(max(d[i - 1] - d[i], 0) for i in range(-n, 0))
        return 100.0 if ls == 0 else 100.0 - 100.0 / (1.0 + g / ls)

    def _consecutive_down(self, c: deque, n: int) -> bool:
        if len(c) < n + 1:
            return False
        d = list(c)
        return all(d[-(i + 1)] < d[-(i + 2)] for i in range(n))

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
            d[0], d[1], d[2] = bar.high, bar.low, bar.close
            sig = self._evaluate(symbol, bar, market_state)
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

    def _evaluate(self, sym: str, bar: Bar, ms: MarketState) -> Signal | None:
        c = self._c[sym]
        h, lo = self._h[sym], self._lo[sym]
        if len(c) < self.min_bars or not self._check_cooldown(sym, bar.time):
            return None

        atr = self._atr(h, lo, c)
        if not atr or atr < 0.01:
            return None

        price = list(c)[-1]

        # Block SHOCK volatility
        snap = ms.regime_snapshot
        vol = str(snap.vol.value).lower() if snap and snap.vol else ""
        if vol == "shock":
            return None

        # Consecutive down days — exhaustion selling
        if not self._consecutive_down(c, self.down_days):
            return None

        # RSI(2) deep oversold
        rsi2 = self._rsi(c, n=2)
        if rsi2 is None or rsi2 >= self.rsi2_threshold:
            return None

        # IBS confirmation — closed in bottom 30% of daily range
        hl, ll = list(h), list(lo)
        day_range = hl[-1] - ll[-1]
        if day_range > 0:
            ibs = (price - ll[-1]) / day_range
            if ibs > self.ibs_threshold:
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
