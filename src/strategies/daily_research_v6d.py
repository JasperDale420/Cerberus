"""Daily Research v6d — Dual-regime daily strategy.

Combines trend-following pullback entries with mean reversion fades.
Uses SMA slope to determine regime, then applies the appropriate logic.
Long-only for robustness. ATR-based stops/targets for consistency.

Iteration 1: Initial implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class dailyresearchv6dStrategy(BaseStrategy):
    name = "daily_research_v6d"
    allow_overnight: bool = True

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.allow_overnight = True

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_bars = int(config.get("min_bars", 50))
        self.sma_fast = int(config.get("sma_fast", 10))
        self.sma_slow = int(config.get("sma_slow", 40))
        self.atr_period = int(config.get("atr_period", 14))
        self.stop_atr = float(config.get("stop_atr", 2.0))
        self.target_atr = float(config.get("target_atr", 3.0))
        # Trend mode: pullback threshold (how close to fast SMA)
        self.pullback_atr = float(config.get("pullback_atr", 1.5))
        # Mean reversion: BB params
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_oversold = float(config.get("rsi_oversold", 30.0))
        # Regime detection: SMA slope threshold
        self.slope_threshold = float(config.get("slope_threshold", 0.002))
        self.allow_overnight = True
        self.max_hold_days = int(config.get("max_hold_days", 10))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars):
            return None

        bars = list(symbol_state.bars)
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        if len(closes) < self.min_bars:
            return None

        price = closes[-1]

        # Calculate indicators
        sma_f = self._sma(closes, self.sma_fast)
        sma_s = self._sma(closes, self.sma_slow)
        atr = self._atr(highs, lows, closes, self.atr_period)

        if sma_f is None or sma_s is None or atr is None or atr < 0.01:
            return None

        # Determine regime via SMA slope (slow SMA over last 10 bars)
        sma_s_prev = self._sma(closes[:-5], self.sma_slow)
        if sma_s_prev is None or sma_s_prev < 0.01:
            return None

        slope = (sma_s - sma_s_prev) / sma_s_prev

        if slope > self.slope_threshold:
            # UPTREND — look for pullback entries
            return self._trend_signal(symbol, bar, market_state, closes, price, sma_f, sma_s, atr)
        elif slope < -self.slope_threshold:
            # DOWNTREND — skip (long-only)
            return None
        else:
            # FLAT — look for mean reversion
            return self._reversion_signal(symbol, bar, market_state, closes, price, atr)

    def _trend_signal(
        self,
        symbol: str,
        bar: Bar,
        market_state: MarketState,
        closes: list,
        price: float,
        sma_f: float,
        sma_s: float,
        atr: float,
    ) -> Optional[Signal]:
        """Trend-following: buy on pullback to fast SMA in uptrend."""
        # Price must be above slow SMA (uptrend confirmed)
        if price < sma_s:
            return None

        # Fast SMA must be above slow SMA
        if sma_f < sma_s:
            return None

        # Price should be near or below fast SMA (pullback)
        distance = price - sma_f
        if distance > atr * self.pullback_atr:
            return None  # Too extended, not a pullback

        # Price shouldn't be too far below (broken trend)
        if distance < -atr * 2.0:
            return None

        # Confirm with recent price action: last bar should show buying
        if len(closes) >= 2 and closes[-1] <= closes[-2]:
            return None  # No buying pressure yet

        stop_price = price - atr * self.stop_atr
        target_price = price + atr * self.target_atr

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={"mode": "TREND_PULLBACK", "atr": round(atr, 2)},
        )

    def _reversion_signal(
        self,
        symbol: str,
        bar: Bar,
        market_state: MarketState,
        closes: list,
        price: float,
        atr: float,
    ) -> Optional[Signal]:
        """Mean reversion: buy at lower Bollinger Band in flat regime."""
        if len(closes) < self.bb_period:
            return None

        bb_mean = self._sma(closes, self.bb_period)
        bb_std = self._std(closes, self.bb_period)
        if bb_mean is None or bb_std is None or bb_std < 0.01:
            return None

        lower_band = bb_mean - self.bb_std * bb_std

        # Only buy if price is below lower band
        if price >= lower_band:
            return None

        # RSI confirmation
        rsi = self._rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_oversold:
            return None

        stop_price = price - atr * self.stop_atr
        target_price = bb_mean  # Target the mean

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol,
            OrderSide.BUY,
            bar,
            market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={"mode": "MEAN_REVERSION", "rsi": round(rsi, 1)},
        )

    # --- Technical indicators ---

    def _sma(self, values: list, period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    def _std(self, values: list, period: int) -> Optional[float]:
        if len(values) < period:
            return None
        data = values[-period:]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance**0.5

    def _rsi(self, values: list, period: int) -> Optional[float]:
        if len(values) < period + 1:
            return None
        gains, losses = [], []
        for i in range(len(values) - period, len(values)):
            change = values[i] - values[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _atr(self, highs: list, lows: list, closes: list, period: int) -> Optional[float]:
        if len(highs) < period + 1:
            return None
        tr_values = []
        for i in range(len(highs) - period, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_values.append(tr)
        if not tr_values:
            return None
        return sum(tr_values) / len(tr_values)
