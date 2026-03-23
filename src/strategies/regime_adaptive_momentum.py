"""
RegimeAdaptiveMomentum — Multi-mode swing trading with internal regime detection.

Solves daily_momentum's melt-up failure by switching entry modes:
  Mode A (Pullback): ADX trending + price pulled back to EMA → buy the dip
  Mode B (Breakout): ADX trending + no pullback → buy new highs
  Mode C (Sit out):  ADX range-bound or extreme volatility → no trades

Uses Wilder's ADX, pullback depth, EMA spread, and ATR ratio to classify
per-symbol, per-day regime internally — no dependency on MarketContextService.
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


class RegimeAdaptiveMomentumStrategy(BaseStrategy):
    """Multi-mode momentum strategy that adapts entries to market regime."""

    name: str = "regime_adaptive_momentum"
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
        # EMA
        self.ema_fast_period = int(config.get("ema_fast_period", 20))
        self.ema_slow_period = int(config.get("ema_slow_period", 50))
        # ADX / regime
        self.adx_period = int(config.get("adx_period", 14))
        self.adx_trend_threshold = float(config.get("adx_trend_threshold", 25.0))
        self.adx_range_threshold = float(config.get("adx_range_threshold", 20.0))
        # Pullback detection
        self.pullback_lookback = int(config.get("pullback_lookback", 10))
        self.pullback_min_pct = float(config.get("pullback_min_pct", 0.02))
        # Volatility
        self.high_vol_mult = float(config.get("high_vol_mult", 2.0))
        self.trade_high_vol = bool(config.get("trade_high_vol", False))
        # Mode A: pullback entries
        self.pullback_entry_pct = float(config.get("pullback_entry_pct", 0.025))
        self.stop_atr_mult_pullback = float(config.get("stop_atr_mult_pullback", 1.5))
        self.target_atr_mult_pullback = float(config.get("target_atr_mult_pullback", 6.0))
        self.confluence_threshold_pullback = float(config.get("confluence_threshold_pullback", 30.0))
        # Mode B: breakout entries
        self.breakout_lookback = int(config.get("breakout_lookback", 20))
        self.breakout_margin_pct = float(config.get("breakout_margin_pct", 0.002))
        self.stop_atr_mult_breakout = float(config.get("stop_atr_mult_breakout", 2.0))
        self.target_atr_mult_breakout = float(config.get("target_atr_mult_breakout", 4.0))
        # General
        self.min_bars = int(config.get("min_bars", 55))
        self.max_hold_days = int(config.get("max_hold_days", 10))
        self.allow_overnight = True
        self.long_only = bool(config.get("long_only", True))
        self.vol_avg_mult = float(config.get("vol_avg_mult", 0.5))
        self.breakout_only = bool(config.get("breakout_only", False))

    # ------------------------------------------------------------------
    # Per-symbol state management
    # ------------------------------------------------------------------

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

    def _adx(self, highs: deque[float], lows: deque[float], closes: deque[float], period: int = 14) -> float | None:
        """Wilder's ADX — measures trend strength (0-100). >25 = trending, <20 = range-bound."""
        n = len(highs)
        if n < 2 * period + 1:
            return None

        h, lo, c = list(highs), list(lows), list(closes)

        # Step 1: compute +DM, -DM, TR for each bar
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

        # Step 2: Wilder's smoothing (initial = sum of first `period` values)
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

        # Step 3: Smooth DX to get ADX
        adx = sum(dx_vals[:period]) / period
        for i in range(period, len(dx_vals)):
            adx = (adx * (period - 1) + dx_vals[i]) / period

        return adx

    def _pullback_depth(self, closes: deque[float], highs: deque[float], lookback: int = 10) -> float:
        """Fraction that current price has pulled back from rolling high (0.0 = at high, 0.05 = 5% below)."""
        recent_highs = list(highs)[-lookback:]
        if not recent_highs:
            return 0.0
        rolling_high = max(recent_highs)
        current = closes[-1]
        if rolling_high <= 0:
            return 0.0
        return max(0.0, (rolling_high - current) / rolling_high)

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def _classify_regime(self, adx: float, pullback: float, atr: float, avg_atr: float, ema_spread: float) -> str:
        """Classify per-symbol regime into one of four modes."""
        # High volatility overrides
        if avg_atr > 0 and atr > avg_atr * self.high_vol_mult:
            return "HIGH_VOL"
        # Range-bound
        if adx < self.adx_range_threshold:
            return "RANGE_BOUND"
        # Trending
        if adx >= self.adx_trend_threshold:
            if pullback >= self.pullback_min_pct:
                return "TRENDING_WITH_PULLBACK"
            else:
                return "TRENDING_NO_PULLBACK"
        # Weak trend (between range and trend thresholds) — treat as range-bound
        return "RANGE_BOUND"

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

        # Regime gating — skip SHOCK
        snapshot = market_state.regime_snapshot
        if snapshot is not None and snapshot.vol == VolRegime.SHOCK:
            return None
        if not self._check_hmm_gate(market_state):
            return None

        # Compute indicators
        ema_fast = self._ema(closes, self.ema_fast_period)
        ema_slow = self._ema(closes, self.ema_slow_period)
        if ema_fast is None or ema_slow is None:
            return None

        price = closes[-1]
        atr = self._atr(highs, lows, closes, self.adx_period)
        if atr is None or atr < 0.01:
            return None

        adx = self._adx(highs, lows, closes, self.adx_period)
        if adx is None:
            return None

        # Average ATR (20-day) for volatility comparison
        atr_values = []
        h_list, l_list, c_list = list(highs), list(lows), list(closes)
        for i in range(max(1, len(c_list) - 20), len(c_list)):
            if i >= 1:
                tr = max(h_list[i] - l_list[i], abs(h_list[i] - c_list[i - 1]), abs(l_list[i] - c_list[i - 1]))
                atr_values.append(tr)
        avg_atr = sum(atr_values) / len(atr_values) if atr_values else atr

        pullback = self._pullback_depth(closes, highs, self.pullback_lookback)
        ema_spread = (ema_fast - ema_slow) / price if price > 0 else 0.0

        # Classify regime
        regime = self._classify_regime(adx, pullback, atr, avg_atr, ema_spread)

        # Volume check
        vol_list = list(volumes)
        avg_vol = sum(vol_list[-20:]) / min(20, len(vol_list[-20:])) if len(vol_list) >= 5 else 0
        current_vol = vol_list[-1] if vol_list else 0

        # Only trade if EMA fast > slow (uptrend) for long-only
        if self.long_only and ema_fast <= ema_slow:
            return None

        # Dispatch by regime
        if regime == "TRENDING_WITH_PULLBACK":
            if not self.breakout_only:
                return self._try_pullback_entry(
                    symbol, bar, price, ema_fast, ema_slow, atr, avg_vol, current_vol, regime
                )
            return None  # Let daily_momentum handle pullbacks
        elif regime == "TRENDING_NO_PULLBACK":
            return self._try_breakout_entry(symbol, bar, price, ema_fast, atr, avg_atr, avg_vol, current_vol, regime)
        elif regime == "HIGH_VOL" and self.trade_high_vol and not self.breakout_only:
            return self._try_pullback_entry(symbol, bar, price, ema_fast, ema_slow, atr, avg_vol, current_vol, regime)
        # RANGE_BOUND or HIGH_VOL (not trading) → sit out
        return None

    # ------------------------------------------------------------------
    # Mode A: Pullback entry
    # ------------------------------------------------------------------

    def _try_pullback_entry(
        self,
        symbol: str,
        bar: Bar,
        price: float,
        ema_fast: float,
        ema_slow: float,
        atr: float,
        avg_vol: float,
        current_vol: float,
        regime: str,
    ) -> Signal | None:
        distance_to_ema = (price - ema_fast) / ema_fast if ema_fast > 0 else 0.0
        closes = self._daily_closes[symbol]

        # Confluence scoring — matches daily_momentum's proven 5-factor system
        score = 0.0

        # Factor 1: Trend strength (EMA spread)
        ema_spread = (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0.0
        if ema_spread > 0.005:
            score += 25.0
        if ema_spread > 0.01:
            score += 10.0

        # Factor 2: Pullback proximity — price near fast EMA
        if -self.pullback_entry_pct <= distance_to_ema <= self.pullback_entry_pct:
            score += 25.0
        elif distance_to_ema < -self.pullback_entry_pct:
            score += 15.0
        else:
            score += 5.0

        # Factor 3: Volume confirmation
        if avg_vol > 0 and current_vol > avg_vol * self.vol_avg_mult:
            score += 20.0

        # Factor 4: Momentum (close > prior close)
        if len(closes) >= 2 and price > closes[-2]:
            score += 15.0

        # Factor 5: Not extended (price not too far from slow EMA)
        distance_to_slow = (price - ema_slow) / ema_slow if ema_slow > 0 else 0.0
        if distance_to_slow < 0.05:
            score += 5.0

        if score >= self.confluence_threshold_pullback:
            stop_price = price - atr * self.stop_atr_mult_pullback
            target_price = price + atr * self.target_atr_mult_pullback
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
                meta={"regime_mode": "PULLBACK", "regime": regime},
            )
        return None

    # ------------------------------------------------------------------
    # Mode B: Breakout entry
    # ------------------------------------------------------------------

    def _try_breakout_entry(
        self,
        symbol: str,
        bar: Bar,
        price: float,
        ema_fast: float,
        atr: float,
        avg_atr: float,
        avg_vol: float,
        current_vol: float,
        regime: str,
    ) -> Signal | None:
        highs = self._daily_highs[symbol]
        closes = self._daily_closes[symbol]
        # Need enough history for breakout lookback
        if len(highs) < self.breakout_lookback + 1:
            return None

        # Rolling high over lookback period (excluding today)
        prior_highs = list(highs)[-(self.breakout_lookback + 1) : -1]
        if not prior_highs:
            return None
        rolling_high = max(prior_highs)

        # Breakout condition: price closes above prior rolling high + margin
        breakout_level = rolling_high * (1.0 + self.breakout_margin_pct)
        if price <= breakout_level:
            return None

        # CONSOLIDATION FILTER: Only buy breakouts from range contraction.
        # If ATR is above average, the market is already volatile/extended —
        # not a clean consolidation breakout.
        if avg_atr > 0 and atr > avg_atr * 1.1:
            return None

        # EXTENSION FILTER: Don't buy if price is already > 5% above slow EMA
        ema_slow = self._ema(closes, self.ema_slow_period)
        if ema_slow and ema_slow > 0:
            extension = (price - ema_slow) / ema_slow
            if extension > 0.08:
                return None

        # Volume confirmation
        if avg_vol > 0 and current_vol <= avg_vol * self.vol_avg_mult:
            return None

        # Price above fast EMA (trend confirmation)
        if price <= ema_fast:
            return None

        stop_price = price - atr * self.stop_atr_mult_breakout
        target_price = price + atr * self.target_atr_mult_breakout
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
            meta={"regime_mode": "BREAKOUT", "regime": regime},
        )
