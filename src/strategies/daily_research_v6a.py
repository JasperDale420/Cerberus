"""Daily Research v6a — RSI(2) mean reversion + EMA trend pullback.

Two complementary signals designed for consistency across all market regimes:
1. RSI(2) mean reversion: Buy oversold dips above SMA50 (works in UP/FLAT/DOWN)
2. EMA pullback: Buy dips to EMA20 in confirmed uptrends (UP/FLAT only)

Minimal parameters to avoid overfitting. ATR-based risk management.
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
        self.target_atr_mult = float(config.get("target_atr_mult", 4.0))
        self.rsi2_threshold = float(config.get("rsi2_threshold", 20))
        self.pullback_dist = float(config.get("pullback_dist", 0.02))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=120)
            self._h[s] = deque(maxlen=120)
            self._lo[s] = deque(maxlen=120)
            self._vol[s] = deque(maxlen=120)
            self._pd[s] = None
            self._dhlcv[s] = [0.0, 0.0, 0.0, 0.0]

    # --- Indicators ---

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

    def _sma(self, vals: list[float], period: int) -> float | None:
        if len(vals) < period:
            return None
        return sum(vals[-period:]) / period

    # --- Bar processing ---

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

    # --- Signal logic ---

    def _evaluate(self, sym: str, bar: Bar, ms: MarketState) -> Signal | None:
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

        # Block SHOCK volatility
        if vol == "shock":
            return None

        # Moving averages
        sma50 = self._sma(cl, 50)
        ema20 = self._ema(cl, 20)
        ema10 = self._ema(cl, 10)

        if ema20 is None or ema10 is None:
            return None

        # Stop/target
        stop_dist = min(atr * self.stop_atr_mult, price * 0.02)
        stop = price - stop_dist
        price + atr * self.target_atr_mult

        # --- Signal 1: RSI(2) Mean Reversion ---
        # Works across ALL regimes. SMA50 gate ensures we buy dips in structurally sound stocks.
        rsi2 = self._rsi(c, n=2)
        if rsi2 is not None and rsi2 < self.rsi2_threshold:
            if sma50 is None or price > sma50:
                self.last_signal_time[sym] = bar.time
                return Signal(
                    symbol=sym,
                    side=OrderSide.BUY,
                    size_hint=1.0,
                    entry_price=price,
                    stop_price=stop,
                    target_price=price + atr * 4.0,
                    strategy=self.name,
                    generated_at=bar.time,
                )

        # Gate: remaining signals only in UP or FLAT
        if trend == "down":
            return None

        # --- Signal 2: EMA Pullback in Uptrend ---
        # Buy when price dips near EMA20 in a confirmed uptrend (EMA10 > EMA20, price > SMA50)
        if ema10 > ema20 and (sma50 is None or price > sma50):
            dist_to_ema = (price - ema20) / ema20
            if -self.pullback_dist <= dist_to_ema <= 0.005:
                # Price is near or just below EMA20 — pullback entry
                self.last_signal_time[sym] = bar.time
                return Signal(
                    symbol=sym,
                    side=OrderSide.BUY,
                    size_hint=1.0,
                    entry_price=price,
                    stop_price=stop,
                    target_price=price + atr * 5.0,
                    strategy=self.name,
                    generated_at=bar.time,
                )

        return None
