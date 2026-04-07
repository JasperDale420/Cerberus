"""Daily Research Strategy — Adaptive Trend + RSI(2) Mean Reversion.

Combines trend-following (SMA) with aggressive RSI(2) dip-buying.
Regime-adaptive: skip SHOCK and DOWN regimes. Long-biased with shorts in downtrends.
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
        self._v: dict[str, deque[float]] = {}
        self._pd: dict[str, date | None] = {}
        self._dhlcv: dict[str, list[float]] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.sma_slow = int(config.get("sma_slow", 50))
        self.rsi_period = int(config.get("rsi_period", 2))
        self.rsi_buy = float(config.get("rsi_buy_threshold", 10.0))
        self.rsi_sell = float(config.get("rsi_sell_threshold", 90.0))
        self.stop_m = float(config.get("stop_atr_mult", 1.5))
        self.tgt_m = float(config.get("target_atr_mult", 3.0))
        self.min_bars = int(config.get("min_bars", 55))
        self.atr_period = int(config.get("atr_period", 14))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            mx = max(self.sma_slow + 10, 80)
            self._c[s] = deque(maxlen=mx)
            self._h[s] = deque(maxlen=mx)
            self._lo[s] = deque(maxlen=mx)
            self._v[s] = deque(maxlen=mx)
            self._pd[s] = None
            self._dhlcv[s] = [0.0, 0.0, 0.0, 0.0]

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

    def _sma(self, v: deque, n: int) -> float | None:
        if len(v) < n:
            return None
        return sum(list(v)[-n:]) / n

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
            d[0], d[1], d[2], d[3] = bar.high, bar.low, bar.close, bar.volume
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

        rsi2 = self._rsi(c, self.rsi_period)
        if rsi2 is None:
            return None

        sma_s = self._sma(c, self.sma_slow)
        if sma_s is None:
            return None

        price = list(c)[-1]

        # Regime info
        snap = ms.regime_snapshot
        trend = str(snap.trend.value).lower() if snap and snap.trend else ""
        vol = str(snap.vol.value).lower() if snap and snap.vol else ""

        # Block SHOCK volatility
        if vol == "shock":
            return None

        # === LONG: RSI(2) oversold + price above slow SMA (uptrend) ===
        if rsi2 < self.rsi_buy and price > sma_s:
            if trend == "down":
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

        # === SHORT: RSI(2) overbought + price below slow SMA (downtrend) ===
        if rsi2 > self.rsi_sell and price < sma_s and trend == "down":
            stop = price + atr * self.stop_m
            target = price - atr * self.tgt_m
            self.last_signal_time[sym] = bar.time
            return Signal(
                symbol=sym,
                side=OrderSide.SELL,
                size_hint=0.0,
                entry_price=price,
                stop_price=stop,
                target_price=target,
                strategy=self.name,
                generated_at=bar.time,
            )

        return None
