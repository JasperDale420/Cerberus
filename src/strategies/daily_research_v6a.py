"""Daily Research v6a — Connors RSI(2) mean reversion with dual SMA filter.

Two-level trend filtering for consistency:
- Index SMA(50): only trade when SPY is above its 50-day SMA (bull market)
- Per-symbol SMA(50): only buy stocks in their own uptrend
- RSI(2) < 10: strict oversold threshold (Connors research)
- Block HIGH and SHOCK volatility regimes
- Tight ATR stops and targets for mean reversion
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
        # Track index (SPY) prices for market-level trend filter
        self._idx_closes: deque[float] = deque(maxlen=250)
        self._idx_pd: date | None = None
        self._idx_last_close: float = 0.0

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 55))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 2.0))
        self.rsi2_threshold = float(config.get("rsi2_threshold", 10))
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

    def _rsi(self, v: deque, n: int = 14) -> float | None:
        if len(v) < n + 1:
            return None
        d = list(v)
        g = sum(max(d[i] - d[i - 1], 0) for i in range(-n, 0))
        ls = sum(max(d[i - 1] - d[i], 0) for i in range(-n, 0))
        return 100.0 if ls == 0 else 100.0 - 100.0 / (1.0 + g / ls)

    def _sma(self, vals, period: int) -> float | None:
        if len(vals) < period:
            return None
        v = list(vals)
        return sum(v[-period:]) / period

    def _update_index(self, bar: Bar) -> None:
        """Track SPY prices for market-level trend filter."""
        dt = bar.time.date() if isinstance(bar.time, datetime) else bar.time
        if self._idx_pd is not None and dt != self._idx_pd:
            self._idx_closes.append(self._idx_last_close)
        self._idx_last_close = bar.close
        self._idx_pd = dt

    def _index_bullish(self) -> bool:
        """Check if the market index is in an uptrend (above SMA50)."""
        if len(self._idx_closes) < self.sma_period:
            return True  # Allow trading during warmup
        idx_sma = self._sma(self._idx_closes, self.sma_period)
        if idx_sma is None:
            return True
        return self._idx_last_close > idx_sma

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # Track index prices
        if symbol.upper() == "SPY":
            self._update_index(bar)

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

        # Block HIGH and SHOCK volatility
        snap = ms.regime_snapshot
        vol = str(snap.vol.value).lower() if snap and snap.vol else ""
        if vol in ("high", "shock"):
            return None

        # Market-level filter: SPY must be above its SMA(50)
        if not self._index_bullish():
            return None

        # Per-symbol SMA(50) uptrend filter
        sma = self._sma(cl, self.sma_period)
        if sma is None or price < sma:
            return None

        # RSI(2) strict oversold — Connors-style
        rsi2 = self._rsi(c, n=2)
        if rsi2 is None:
            return None

        if rsi2 < self.rsi2_threshold:
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
