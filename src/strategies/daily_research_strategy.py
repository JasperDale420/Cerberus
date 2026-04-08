"""Daily Research Strategy — concentrated signals with strong trend filters."""

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
        self._vol: dict[str, deque[float]] = {}
        self._pd: dict[str, date | None] = {}
        self._dhlcv: dict[str, list[float]] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.stop_m = float(config.get("stop_atr_mult", 1.5))
        self.tgt_m = float(config.get("target_atr_mult", 8.0))
        self.rsi2_threshold = float(config.get("rsi2_threshold", 10.0))
        self.breakout_period = int(config.get("breakout_period", 20))
        self.min_bars = int(config.get("min_bars", 55))
        self.vol_mult = float(config.get("vol_mult", 1.2))
        self.roc_period = int(config.get("roc_period", 20))
        self.allow_overnight = True

    def _init(self, s: str) -> None:
        if s not in self._c:
            self._c[s] = deque(maxlen=100)
            self._h[s] = deque(maxlen=100)
            self._lo[s] = deque(maxlen=100)
            self._vol[s] = deque(maxlen=100)
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

        # Hard gates: skip DOWN trend and SHOCK/HIGH volatility
        if trend == "down" or vol in ("shock", "high"):
            return None

        # Strong trend filters
        sma20 = sum(cl[-20:]) / 20
        sma50 = sum(cl[-50:]) / 50 if len(cl) >= 50 else None
        ema10 = self._ema(cl, 10)
        ema20 = self._ema(cl, 20)

        if sma50 is None or ema10 is None or ema20 is None:
            return None

        # Require price above both SMAs (strong uptrend)
        if price <= sma20 or price <= sma50:
            return None

        # Require positive rate of change (momentum confirmation)
        roc_p = self.roc_period
        if len(cl) > roc_p and cl[-roc_p - 1] > 0:
            roc = (price - cl[-roc_p - 1]) / cl[-roc_p - 1]
            if roc <= 0:
                return None
        else:
            return None

        # Regime-adaptive position sizing
        if trend == "up" and vol in ("low", "normal"):
            size = 1.0
        elif trend == "flat" and vol in ("low", "normal"):
            size = 0.6
        else:
            size = 0.3

        # --- Signal 1: RSI(2) Extreme Mean Reversion ---
        rsi2 = self._rsi(c, n=2)
        if rsi2 is not None and rsi2 < self.rsi2_threshold:
            stop = price - atr * self.stop_m
            target = price + atr * 4.0
            self.last_signal_time[sym] = bar.time
            return Signal(
                symbol=sym,
                side=OrderSide.BUY,
                size_hint=size,
                entry_price=price,
                stop_price=stop,
                target_price=target,
                strategy=self.name,
                generated_at=bar.time,
            )

        # --- Signal 2: Momentum Breakout (UP trend only) ---
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
            and sma20 > sma50
            and (avg_vol == 0 or cur_vol > avg_vol * self.vol_mult)
        ):
            stop = price - atr * 2.0
            target = price + atr * self.tgt_m
            self.last_signal_time[sym] = bar.time
            return Signal(
                symbol=sym,
                side=OrderSide.BUY,
                size_hint=size,
                entry_price=price,
                stop_price=stop,
                target_price=target,
                strategy=self.name,
                generated_at=bar.time,
            )

        return None
