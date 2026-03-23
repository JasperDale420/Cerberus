"""
DailyMeanReversion — Daily Bollinger Band fade for range-bound markets.

Type: mean-reversion / counter-trend
Description: Fades extreme deviations from the 20-day mean when ADX confirms
a range-bound market (ADX < threshold). Complements momentum strategies by
capturing profit in flat/choppy regimes where momentum sits out.

Signal Logic:
    LONG when:
    1. ADX < range_threshold (range-bound market confirmed)
    2. Price < lower Bollinger Band (oversold)
    3. RSI(14) < rsi_oversold (momentum extreme)
    4. Volume above minimum threshold (participation)

    SHORT when (if allow_short=True):
    1. ADX < range_threshold
    2. Price > upper Bollinger Band (overbought)
    3. RSI(14) > rsi_overbought
    4. Volume above minimum threshold

Exits: Target = BB midline, Stop = 1.5σ beyond entry band.
Hold period: up to max_hold_days (default 5).
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


class DailyMeanReversionStrategy(BaseStrategy):
    """Bollinger Band mean reversion on daily bars for range-bound markets."""

    name: str = "daily_mean_reversion"
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
        # Bollinger Bands
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        # ADX regime filter
        self.adx_period = int(config.get("adx_period", 14))
        self.adx_range_threshold = float(config.get("adx_range_threshold", 22.0))
        # RSI confirmation
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_oversold = float(config.get("rsi_oversold", 30.0))
        self.rsi_overbought = float(config.get("rsi_overbought", 70.0))
        # Stop/target
        self.stop_bb_mult = float(config.get("stop_bb_mult", 1.5))
        # General
        self.min_bars = int(config.get("min_bars", 30))
        self.max_hold_days = int(config.get("max_hold_days", 5))
        self.allow_overnight = True
        self.allow_short = bool(config.get("allow_short", False))
        self.vol_avg_mult = float(config.get("vol_avg_mult", 0.5))
        # When True, use MarketContextService trend=FLAT filter (aligned with activation policies).
        # When False, fall back to internal ADX filter.
        self.use_regime_trend_filter = bool(config.get("use_regime_trend_filter", True))

    # ------------------------------------------------------------------
    # Per-symbol state management (same pattern as daily_momentum)
    # ------------------------------------------------------------------

    def _ensure_symbol(self, symbol: str) -> None:
        if symbol not in self._daily_closes:
            lookback = max(self.bb_period + 10, 40)
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

    def _sma(self, values: deque[float], period: int) -> float | None:
        if len(values) < period:
            return None
        return sum(list(values)[-period:]) / period

    def _std(self, values: deque[float], period: int) -> float | None:
        if len(values) < period:
            return None
        data = list(values)[-period:]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance**0.5

    def _rsi(self, values: deque[float], period: int) -> float | None:
        if len(values) < period + 1:
            return None
        data = list(values)
        gains, losses = [], []
        for i in range(len(data) - period, len(data)):
            change = data[i] - data[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _adx(self, highs: deque[float], lows: deque[float], closes: deque[float], period: int = 14) -> float | None:
        """Wilder's ADX — same implementation as regime_adaptive_momentum."""
        n = len(highs)
        if n < 2 * period + 1:
            return None
        h, lo, c = list(highs), list(lows), list(closes)
        plus_dm_vals: list[float] = []
        minus_dm_vals: list[float] = []
        tr_vals: list[float] = []
        for i in range(1, n):
            up_move = h[i] - h[i - 1]
            down_move = lo[i - 1] - lo[i]
            plus_dm_vals.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
            minus_dm_vals.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
            tr_vals.append(max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])))
        if len(tr_vals) < 2 * period:
            return None
        smoothed_plus_dm = sum(plus_dm_vals[:period])
        smoothed_minus_dm = sum(minus_dm_vals[:period])
        smoothed_tr = sum(tr_vals[:period])
        dx_vals: list[float] = []
        for i in range(period, len(tr_vals)):
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm_vals[i]
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm_vals[i]
            smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr_vals[i]
            if smoothed_tr > 0:
                plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
                minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
            else:
                plus_di = minus_di = 0.0
            di_sum = plus_di + minus_di
            dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
            dx_vals.append(dx)
        if len(dx_vals) < period:
            return None
        adx = sum(dx_vals[:period]) / period
        for i in range(period, len(dx_vals)):
            adx = (adx * (period - 1) + dx_vals[i]) / period
        return adx

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
        volumes = self._daily_volumes[symbol]

        if len(closes) < self.min_bars:
            return None
        if not self._check_cooldown(symbol, bar.time):
            return None

        # Regime gating — skip SHOCK volatility
        snapshot = market_state.regime_snapshot
        if snapshot is not None and snapshot.vol == VolRegime.SHOCK:
            return None

        # ADX filter — only trade in range-bound markets
        # Note: MarketContextService's Hurst-based trend is not well-calibrated for daily bars
        # (needs intraday data), so we use internal ADX as the primary range-bound filter.
        adx = self._adx(highs, lows, closes, self.adx_period)
        if adx is not None and adx >= self.adx_range_threshold:
            return None  # Market is trending — not our regime

        # Bollinger Bands
        bb_mean = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mean is None or bb_std is None or bb_std < 0.01:
            return None

        upper_band = bb_mean + self.bb_std * bb_std
        lower_band = bb_mean - self.bb_std * bb_std
        price = closes[-1]

        # RSI confirmation
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None:
            return None

        # Volume check
        vol_list = list(volumes)
        avg_vol = sum(vol_list[-20:]) / min(20, len(vol_list[-20:])) if len(vol_list) >= 5 else 0
        current_vol = vol_list[-1] if vol_list else 0

        # LONG: price below lower band + RSI oversold
        if price < lower_band and rsi < self.rsi_oversold:
            if avg_vol > 0 and current_vol < avg_vol * self.vol_avg_mult:
                return None  # Not enough volume

            stop_price = price - self.stop_bb_mult * bb_std
            target_price = bb_mean  # Target the mean
            z_score = (price - bb_mean) / bb_std if bb_std > 0 else 0.0

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
                    "mode": "LONG_REVERSION",
                    "z_score": round(z_score, 2),
                    "adx": round(adx or 0, 1),
                    "rsi": round(rsi, 1),
                },
            )

        # SHORT: price above upper band + RSI overbought
        if self.allow_short and price > upper_band and rsi > self.rsi_overbought:
            if avg_vol > 0 and current_vol < avg_vol * self.vol_avg_mult:
                return None

            stop_price = price + self.stop_bb_mult * bb_std
            target_price = bb_mean
            z_score = (price - bb_mean) / bb_std if bb_std > 0 else 0.0

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
                meta={
                    "mode": "SHORT_REVERSION",
                    "z_score": round(z_score, 2),
                    "adx": round(adx or 0, 1),
                    "rsi": round(rsi, 1),
                },
            )

        return None
