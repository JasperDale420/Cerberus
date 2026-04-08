"""Daily Research Strategy — vol-targeted sizing, long+short, multi-signal."""

from __future__ import annotations

import math
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
        self._vol: dict[str, deque[float]] = {}
        self._pd: dict[str, date | None] = {}
        self._dhlcv: dict[str, list[float]] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.stop_m = float(config.get("stop_atr_mult", 1.5))
        self.tgt_m = float(config.get("target_atr_mult", 6.0))
        self.rsi2_threshold = float(config.get("rsi2_threshold", 15.0))
        self.rsi2_short_threshold = float(config.get("rsi2_short_threshold", 85.0))
        self.breakout_period = int(config.get("breakout_period", 20))
        self.min_bars = int(config.get("min_bars", 25))
        self.pullback_rsi_lo = float(config.get("pullback_rsi_lo", 30.0))
        self.pullback_rsi_hi = float(config.get("pullback_rsi_hi", 55.0))
        self.vol_mult = float(config.get("vol_mult", 1.0))
        self.vol_target = float(config.get("vol_target", 0.15))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=80)
            self._h[s] = deque(maxlen=80)
            self._lo[s] = deque(maxlen=80)
            self._vol[s] = deque(maxlen=80)
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

    def _ema(self, vals: list[float], period: int) -> float | None:
        if len(vals) < period:
            return None
        k = 2.0 / (period + 1)
        ema = vals[0]
        for v in vals[1:]:
            ema = v * k + ema * (1 - k)
        return ema

    def _realized_vol(self, closes: list[float], lookback: int = 20) -> float | None:
        if len(closes) < lookback + 1:
            return None
        rets = [(closes[i] / closes[i - 1]) - 1 for i in range(-lookback, 0)]
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var * 252) if var > 0 else None

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
            sig = self._sig(symbol, bar, market_state)
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

    def _sig(self, sym: str, bar: Bar, ms: MarketState) -> Signal | None:
        c = self._c[sym]
        if len(c) < self.min_bars or not self._check_cooldown(sym, bar.time):
            return None
        atr = self._atr(self._h[sym], self._lo[sym], c)
        if not atr or atr < 0.01:
            return None

        cl = list(c)
        price = cl[-1]

        # Regime info
        snap = ms.regime_snapshot
        trend = str(snap.trend.value).lower() if snap and snap.trend else ""
        vol = str(snap.vol.value).lower() if snap and snap.vol else ""

        # SHOCK volatility: skip all signals
        if vol == "shock":
            return None

        # Moving averages
        sma20 = sum(cl[-20:]) / 20 if len(cl) >= 20 else None
        ema10 = self._ema(cl, 10)
        ema20 = self._ema(cl, 20)

        if sma20 is None or ema10 is None or ema20 is None:
            return None

        # Vol-targeted sizing
        rv = self._realized_vol(cl)
        if rv and rv > 0.01:
            size = min(self.vol_target / rv, 1.0)
        elif trend == "up":
            size = 0.8
        else:
            size = 0.5
        if trend == "up":
            size = min(size * 1.2, 1.0)

        # === DOWN REGIME: SHORT SIGNALS ===
        if trend == "down":
            return self._short_signals(sym, bar, cl, price, atr, sma20, ema10, ema20, size * 0.5)

        # === UP/FLAT REGIME: LONG SIGNALS ===
        return self._long_signals(sym, bar, cl, price, atr, sma20, ema10, ema20, size, trend)

    def _long_signals(
        self,
        sym: str,
        bar: Bar,
        cl: list,
        price: float,
        atr: float,
        sma20: float,
        ema10: float,
        ema20: float,
        size: float,
        trend: str,
    ) -> Signal | None:
        c = self._c[sym]

        # Signal 1: RSI(2) Mean Reversion
        rsi2 = self._rsi(c, n=2)
        if rsi2 is not None and rsi2 < self.rsi2_threshold and price > sma20:
            self.last_signal_time[sym] = bar.time
            return Signal(
                symbol=sym,
                side=OrderSide.BUY,
                size_hint=size,
                entry_price=price,
                stop_price=price - atr * self.stop_m,
                target_price=price + atr * 4.0,
                strategy=self.name,
                generated_at=bar.time,
            )

        # Signal 2: Momentum Breakout (UP only)
        bp = self.breakout_period
        prev = cl[-(bp + 1) : -1] if len(cl) > bp else []
        vol_list = list(self._vol[sym])
        avg_vol = sum(vol_list[-20:]) / min(20, len(vol_list[-20:])) if len(vol_list) >= 5 else 0
        cur_vol = vol_list[-1] if vol_list else 0

        if (
            trend == "up"
            and len(prev) >= bp
            and price > max(prev)
            and ema10 > ema20
            and (avg_vol == 0 or cur_vol > avg_vol * self.vol_mult)
        ):
            self.last_signal_time[sym] = bar.time
            return Signal(
                symbol=sym,
                side=OrderSide.BUY,
                size_hint=size,
                entry_price=price,
                stop_price=price - atr * 2.0,
                target_price=price + atr * self.tgt_m,
                strategy=self.name,
                generated_at=bar.time,
            )

        # Signal 3: Pullback in Uptrend
        rsi14 = self._rsi(c, n=14)
        if (
            rsi14 is not None
            and self.pullback_rsi_lo <= rsi14 <= self.pullback_rsi_hi
            and ema10 > ema20
            and price > sma20
            and price <= ema10
        ):
            self.last_signal_time[sym] = bar.time
            return Signal(
                symbol=sym,
                side=OrderSide.BUY,
                size_hint=size * 0.8,
                entry_price=price,
                stop_price=price - atr * self.stop_m,
                target_price=price + atr * 5.0,
                strategy=self.name,
                generated_at=bar.time,
            )

        return None

    def _short_signals(
        self,
        sym: str,
        bar: Bar,
        cl: list,
        price: float,
        atr: float,
        sma20: float,
        ema10: float,
        ema20: float,
        size: float,
    ) -> Signal | None:
        c = self._c[sym]

        # Short Signal 1: RSI(2) Overbought in downtrend
        rsi2 = self._rsi(c, n=2)
        if rsi2 is not None and rsi2 > self.rsi2_short_threshold and price < sma20:
            self.last_signal_time[sym] = bar.time
            return Signal(
                symbol=sym,
                side=OrderSide.SELL,
                size_hint=size,
                entry_price=price,
                stop_price=price + atr * self.stop_m,
                target_price=price - atr * 4.0,
                strategy=self.name,
                generated_at=bar.time,
            )

        # Short Signal 2: Breakdown below N-day low
        bp = self.breakout_period
        prev = cl[-(bp + 1) : -1] if len(cl) > bp else []
        if len(prev) >= bp and price < min(prev) and ema10 < ema20:
            self.last_signal_time[sym] = bar.time
            return Signal(
                symbol=sym,
                side=OrderSide.SELL,
                size_hint=size,
                entry_price=price,
                stop_price=price + atr * 2.0,
                target_price=price - atr * self.tgt_m,
                strategy=self.name,
                generated_at=bar.time,
            )

        return None
