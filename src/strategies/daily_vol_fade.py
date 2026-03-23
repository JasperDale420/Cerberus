"""
DailyVolFade — Crisis alpha: buy extreme daily selloffs.

Type: volatility mean-reversion / crisis alpha
Description: Buys when a stock has an extreme down day (close falls > 2 ATR
below the 20-day EMA) during elevated volatility. Captures the bounce as
panic selling is absorbed. Complements momentum strategies by profiting
in high-volatility regimes where momentum sits out.

Signal Logic:
    LONG when:
    1. Close < EMA(20) - (deviation_mult × ATR(14)) — extreme selloff
    2. ATR(14) > vol_expansion_mult × 20-day avg ATR — volatility expansion
    3. Price above a minimum floor (not penny stock collapse)

Exits: Target = 50% retracement toward EMA(20), Stop = day_low - 0.5 × ATR.
Hold period: 1-3 days max (quick bounce capture).
Long only — no shorting extreme rallies (momentum is too strong).
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from typing import Any

from src.core.domain import (
    Bar,
    MarketState,
    OrderSide,
    Signal,
    SymbolState,
    VolRegime,
)
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class DailyVolFadeStrategy(BaseStrategy):
    """Buy extreme daily selloffs during volatility expansion."""

    name: str = "daily_vol_fade"
    allow_overnight: bool = True

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)
        self._daily_closes: dict[str, deque[float]] = {}
        self._daily_highs: dict[str, deque[float]] = {}
        self._daily_lows: dict[str, deque[float]] = {}
        self._daily_volumes: dict[str, deque[float]] = {}
        self._last_bar_date: dict[str, date | None] = {}
        self._intraday_high: dict[str, float] = {}
        self._intraday_low: dict[str, float] = {}
        self._intraday_volume: dict[str, float] = {}
        self._intraday_close: dict[str, float] = {}

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        # EMA for mean reference
        self.ema_period = int(config.get("ema_period", 20))
        # ATR for volatility measurement
        self.atr_period = int(config.get("atr_period", 14))
        # Entry: price must be this many ATR below EMA
        self.deviation_mult = float(config.get("deviation_mult", 2.0))
        # Volatility expansion: current ATR must be > this × avg ATR
        self.vol_expansion_mult = float(config.get("vol_expansion_mult", 1.5))
        # Target: retracement fraction toward EMA
        self.retracement_pct = float(config.get("retracement_pct", 0.5))
        # Stop: distance below day low in ATR units
        self.stop_atr_below_low = float(config.get("stop_atr_below_low", 0.5))
        # General
        self.min_bars = int(config.get("min_bars", 25))
        self.max_hold_days = int(config.get("max_hold_days", 3))
        self.allow_overnight = True
        self.min_price = float(config.get("min_price", 5.0))
        # When True, require MarketContextService to classify vol as HIGH or SHOCK.
        # When False, rely only on internal ATR-based vol expansion check.
        self.require_high_vol_regime = bool(config.get("require_high_vol_regime", False))

    # ------------------------------------------------------------------
    # Per-symbol state management (same pattern as daily_momentum)
    # ------------------------------------------------------------------

    def _ensure_symbol(self, symbol: str) -> None:
        if symbol not in self._daily_closes:
            lookback = max(self.ema_period + 10, 40)
            self._daily_closes[symbol] = deque(maxlen=lookback)
            self._daily_highs[symbol] = deque(maxlen=lookback)
            self._daily_lows[symbol] = deque(maxlen=lookback)
            self._daily_volumes[symbol] = deque(maxlen=lookback)
            self._last_bar_date[symbol] = None
            self._intraday_high[symbol] = 0.0
            self._intraday_low[symbol] = float("inf")
            self._intraday_volume[symbol] = 0.0
            self._intraday_close[symbol] = 0.0

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    def _ema(self, values: deque[float], period: int) -> float | None:
        if len(values) < period:
            return None
        mult = 2.0 / (period + 1)
        ema = values[0]
        for v in list(values)[1:]:
            ema = v * mult + ema * (1 - mult)
        return ema

    def _atr(self, highs: deque[float], lows: deque[float], closes: deque[float], period: int = 14) -> float | None:
        if len(highs) < period + 1:
            return None
        h, lo, c = list(highs), list(lows), list(closes)
        trs = []
        for i in range(1, min(period + 1, len(h))):
            tr = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else None

    # ------------------------------------------------------------------
    # Bar processing (same accumulation pattern as daily_momentum)
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        self._ensure_symbol(symbol)
        current_date = bar.time.date() if isinstance(bar.time, datetime) else bar.time

        if self._last_bar_date[symbol] is not None and current_date != self._last_bar_date[symbol]:
            # New day — save previous day, evaluate signal
            self._daily_closes[symbol].append(self._intraday_close[symbol])
            self._daily_highs[symbol].append(self._intraday_high[symbol])
            self._daily_lows[symbol].append(self._intraday_low[symbol])
            self._daily_volumes[symbol].append(self._intraday_volume[symbol])

            self._intraday_high[symbol] = bar.high
            self._intraday_low[symbol] = bar.low
            self._intraday_volume[symbol] = bar.volume
            self._intraday_close[symbol] = bar.close

            signal = self._evaluate_daily_signal(symbol, bar, symbol_state, market_state)
            self._last_bar_date[symbol] = current_date
            return signal
        else:
            # Same day — accumulate
            if self._last_bar_date[symbol] is None:
                self._intraday_high[symbol] = bar.high
                self._intraday_low[symbol] = bar.low
                self._intraday_volume[symbol] = bar.volume
            else:
                self._intraday_high[symbol] = max(self._intraday_high[symbol], bar.high)
                self._intraday_low[symbol] = min(self._intraday_low[symbol], bar.low)
                self._intraday_volume[symbol] += bar.volume
            self._intraday_close[symbol] = bar.close
            self._last_bar_date[symbol] = current_date
            return None

    # ------------------------------------------------------------------
    # Daily signal evaluation
    # ------------------------------------------------------------------

    def _evaluate_daily_signal(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        closes = self._daily_closes[symbol]
        highs = self._daily_highs[symbol]
        lows = self._daily_lows[symbol]

        if len(closes) < self.min_bars:
            return None
        if not self._check_cooldown(symbol, bar.time):
            return None

        # Optionally require HIGH/SHOCK vol regime from MarketContextService
        snapshot = market_state.regime_snapshot
        if self.require_high_vol_regime and snapshot is not None:
            if snapshot.vol not in (VolRegime.HIGH, VolRegime.SHOCK):
                return None

        price = closes[-1]
        if price < self.min_price:
            return None

        # EMA(20) — the mean we expect price to revert to
        ema = self._ema(closes, self.ema_period)
        if ema is None:
            return None

        # ATR(14) — current volatility
        atr = self._atr(highs, lows, closes, self.atr_period)
        if atr is None or atr < 0.01:
            return None

        # Average ATR (20-day) for volatility expansion check
        h_list, l_list, c_list = list(highs), list(lows), list(closes)
        atr_values = []
        for i in range(max(1, len(c_list) - 20), len(c_list)):
            if i >= 1:
                tr = max(h_list[i] - l_list[i], abs(h_list[i] - c_list[i - 1]), abs(l_list[i] - c_list[i - 1]))
                atr_values.append(tr)
        avg_atr = sum(atr_values) / len(atr_values) if atr_values else atr

        # Check 1: Extreme selloff — price far below EMA
        deviation_threshold = ema - (self.deviation_mult * atr)
        if price >= deviation_threshold:
            return None  # Not extreme enough

        # Check 2: Volatility expansion — ATR above average
        if avg_atr > 0 and atr < avg_atr * self.vol_expansion_mult:
            return None  # Volatility not elevated enough

        # Passed all filters — generate buy signal
        day_low = lows[-1]
        stop_price = day_low - self.stop_atr_below_low * atr
        # Target: partial retracement toward EMA
        distance_to_ema = ema - price
        target_price = price + distance_to_ema * self.retracement_pct

        deviation_pct = (ema - price) / ema if ema > 0 else 0.0

        self.last_signal_time[symbol] = bar.time
        return Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            size_hint=0.0,
            entry_price=price,
            stop_price=stop_price,
            target_price=target_price,
            strategy=self.name,
            generated_at=bar.time,
            meta={
                "mode": "VOL_FADE",
                "deviation_pct": round(deviation_pct * 100, 2),
                "atr_ratio": round(atr / avg_atr, 2) if avg_atr > 0 else 0.0,
            },
        )
