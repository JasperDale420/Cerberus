"""
DailyMomentum Strategy — Simple Swing Trading

Type: trend-following / momentum
Description: Captures multi-day trends using simple moving average crossovers.
Designed for 5-minute bar input but operates on daily close-to-close price action.

Signal Logic:
    LONG when:
    1. Price is above 20-day EMA (uptrend confirmed)
    2. 20-day EMA > 50-day EMA (momentum)
    3. Today's close pulls back near the 20 EMA (entry on dip)
    4. Volume is above average (participation)

    SHORT when:
    1. Price is below 20-day EMA (downtrend confirmed)
    2. 20-day EMA < 50-day EMA (momentum)
    3. Today's close bounces near the 20 EMA (entry on rally)
    4. Volume is above average

Exits: Trailing stop based on ATR, or trend reversal (EMA crossover flip).
Hold period: up to max_hold_days (default 10).
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


class DailyMomentumStrategy(BaseStrategy):
    """Simple daily momentum strategy for swing trading."""

    name: str = "daily_momentum"
    allow_overnight: bool = True

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)
        # Per-symbol daily tracking
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
        self.ema_fast_period = int(config.get("ema_fast_period", 20))
        self.ema_slow_period = int(config.get("ema_slow_period", 50))
        self.pullback_pct = float(config.get("pullback_pct", 0.015))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 2.0))
        self.target_atr_mult = float(config.get("target_atr_mult", 4.0))
        self.min_bars = int(config.get("min_bars", 55))
        self.max_hold_days = int(config.get("max_hold_days", 10))
        self.allow_overnight = True
        self.confluence_threshold = float(config.get("confluence_threshold", 50.0))
        self.long_only = bool(config.get("long_only", True))
        self.vol_avg_mult = float(config.get("vol_avg_mult", 0.8))

    def _ensure_symbol(self, symbol: str) -> None:
        if symbol not in self._daily_closes:
            lookback = max(self.ema_slow_period + 10, 60)
            self._daily_closes[symbol] = deque(maxlen=lookback)
            self._daily_highs[symbol] = deque(maxlen=lookback)
            self._daily_lows[symbol] = deque(maxlen=lookback)
            self._daily_volumes[symbol] = deque(maxlen=lookback)
            self._last_bar_date[symbol] = None
            self._intraday_high[symbol] = 0.0
            self._intraday_low[symbol] = float("inf")
            self._intraday_volume[symbol] = 0.0
            self._intraday_close[symbol] = 0.0

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
        tr_values = []
        closes_list = list(closes)
        highs_list = list(highs)
        lows_list = list(lows)
        for i in range(1, min(period + 1, len(highs_list))):
            tr = max(
                highs_list[i] - lows_list[i],
                abs(highs_list[i] - closes_list[i - 1]),
                abs(lows_list[i] - closes_list[i - 1]),
            )
            tr_values.append(tr)
        if not tr_values:
            return None
        return sum(tr_values) / len(tr_values)

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        """Process bars — handles both intraday aggregation and direct daily bars."""
        self._ensure_symbol(symbol)

        current_date = bar.time.date() if isinstance(bar.time, datetime) else bar.time

        # Check if we've moved to a new day
        if self._last_bar_date[symbol] is not None and current_date != self._last_bar_date[symbol]:
            # Close out the previous day — save daily bar
            self._daily_closes[symbol].append(self._intraday_close[symbol])
            self._daily_highs[symbol].append(self._intraday_high[symbol])
            self._daily_lows[symbol].append(self._intraday_low[symbol])
            self._daily_volumes[symbol].append(self._intraday_volume[symbol])

            # Reset intraday accumulators for new day
            self._intraday_high[symbol] = bar.high
            self._intraday_low[symbol] = bar.low
            self._intraday_volume[symbol] = bar.volume
            self._intraday_close[symbol] = bar.close

            # Evaluate signal on the new bar (using accumulated daily data)
            signal = self._evaluate_daily_signal(symbol, bar, symbol_state, market_state)
            self._last_bar_date[symbol] = current_date
            return signal
        else:
            # Same day — update intraday accumulators
            if self._last_bar_date[symbol] is None:
                # First bar ever for this symbol
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

    def _evaluate_daily_signal(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        """Evaluate signal based on daily aggregated data."""
        closes = self._daily_closes[symbol]
        highs = self._daily_highs[symbol]
        lows = self._daily_lows[symbol]
        volumes = self._daily_volumes[symbol]

        if len(closes) < self.min_bars:
            return None

        # Cooldown
        if not self._check_cooldown(symbol, bar.time):
            return None

        # Regime gating — skip SHOCK and HIGH volatility
        # Empirical: -$22K in SHOCK, -$34K in HIGH vol (2020-2025 regime analysis)
        snapshot = market_state.regime_snapshot
        if snapshot is not None and snapshot.vol in (VolRegime.SHOCK, VolRegime.HIGH):
            return None

        # HMM gate
        if not self._check_hmm_gate(market_state):
            return None

        # Calculate EMAs
        ema_fast = self._ema(closes, self.ema_fast_period)
        ema_slow = self._ema(closes, self.ema_slow_period)

        if ema_fast is None or ema_slow is None:
            return None

        price = closes[-1]
        atr = self._atr(highs, lows, closes)
        if atr is None or atr < 0.01:
            return None

        # Volume check
        vol_list = list(volumes)
        avg_vol = sum(vol_list[-20:]) / min(20, len(vol_list[-20:])) if len(vol_list) >= 5 else 0
        current_vol = vol_list[-1] if vol_list else 0

        # === LONG SIGNAL ===
        if ema_fast > ema_slow:
            # Uptrend confirmed
            # Check for pullback to fast EMA
            distance_to_ema = (price - ema_fast) / ema_fast

            # Confluence scoring
            score = 0.0

            # Factor 1: Trend strength (EMA spread)
            ema_spread = (ema_fast - ema_slow) / ema_slow
            if ema_spread > 0.005:
                score += 25.0
            if ema_spread > 0.01:
                score += 10.0

            # Factor 2: Pullback proximity (price near fast EMA)
            if -self.pullback_pct <= distance_to_ema <= self.pullback_pct:
                score += 25.0  # Near EMA — ideal pullback entry
            elif distance_to_ema < -self.pullback_pct:
                score += 15.0  # Below EMA — potential oversold bounce
            else:
                score += 5.0  # Above EMA — chase entry (penalized)

            # Factor 3: Volume confirmation
            if avg_vol > 0 and current_vol > avg_vol * self.vol_avg_mult:
                score += 20.0

            # Factor 4: Price above prior day's close (momentum confirmation)
            if len(closes) >= 2 and price > closes[-2]:
                score += 15.0

            # Factor 5: Not extended (price not too far from slow EMA)
            distance_to_slow = (price - ema_slow) / ema_slow
            if distance_to_slow < 0.05:  # Not more than 5% above slow EMA
                score += 5.0

            if score >= self.confluence_threshold:
                stop_price = price - atr * self.stop_atr_mult
                target_price = price + atr * self.target_atr_mult
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
                )

        # === SHORT SIGNAL (if enabled) ===
        if not self.long_only and ema_fast < ema_slow:
            distance_to_ema = (ema_fast - price) / ema_fast

            score = 0.0
            ema_spread = (ema_slow - ema_fast) / ema_slow
            if ema_spread > 0.005:
                score += 25.0
            if ema_spread > 0.01:
                score += 10.0

            if -self.pullback_pct <= distance_to_ema <= self.pullback_pct:
                score += 25.0
            elif distance_to_ema < -self.pullback_pct:
                score += 15.0
            else:
                score += 5.0

            if avg_vol > 0 and current_vol > avg_vol * self.vol_avg_mult:
                score += 20.0

            if len(closes) >= 2 and price < closes[-2]:
                score += 15.0

            distance_to_slow = (ema_slow - price) / ema_slow
            if distance_to_slow < 0.05:
                score += 5.0

            if score >= self.confluence_threshold:
                stop_price = price + atr * self.stop_atr_mult
                target_price = price - atr * self.target_atr_mult
                self.last_signal_time[symbol] = bar.time
                return Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    size_hint=0.0,
                    entry_price=price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy=self.name,
                    generated_at=bar.time,
                )

        return None
