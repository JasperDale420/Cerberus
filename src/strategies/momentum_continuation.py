"""
Momentum Continuation Strategy.

Rides initial breakout momentum without waiting for pullbacks.
Captures the first leg of breakouts that pullback strategies miss.
"""

from __future__ import annotations

from datetime import time as time_type
from typing import Any, Dict, List, Optional

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.data.calculator import FeatureCalculator
from src.strategies.base import BaseStrategy


class MomentumContinuationStrategy(BaseStrategy):
    """
    Momentum Continuation Strategy.

    Enters on breakouts of multi-day highs/lows with volume confirmation.
    Does NOT wait for pullbacks - captures the initial momentum leg.

    Activation: trend=up/down, vol=normal/high, session=opening
    """

    name = "momentum_continuation"

    # Default thresholds
    DEFAULT_BREAKOUT_LOOKBACK = 5  # Days for high/low
    DEFAULT_VOL_MULT = 2.0  # Volume > 2x average
    DEFAULT_CLOSE_POSITION = 0.75  # Close in top 25% of range
    DEFAULT_RISK_REWARD = 1.5

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)

        # Breakout parameters
        self.breakout_lookback = int(
            config.get("breakout_lookback", self.DEFAULT_BREAKOUT_LOOKBACK)
        )
        self.vol_mult = float(config.get("vol_mult", self.DEFAULT_VOL_MULT))
        self.close_position_threshold = float(
            config.get("close_position", self.DEFAULT_CLOSE_POSITION)
        )
        self.risk_reward = float(config.get("risk_reward", self.DEFAULT_RISK_REWARD))

        # EMA for trend confirmation
        self.ema_fast = int(config.get("ema_fast", 20))
        self.ema_slow = int(config.get("ema_slow", 50))

        # Time window (morning momentum)
        self.entry_start = time_type(9, 35)
        self.entry_end = time_type(11, 0)

        # Position management
        self.max_trades_per_session = int(config.get("max_trades_per_session", 2))
        self._session_trades: Dict[str, int] = {}
        self._traded_symbols_today: set = set()
        self._last_date: Optional[str] = None

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # 1. Regime check (trending only)
        if market_state.regime not in (Regime.BULL, Regime.BEAR):
            return None

        # 2. Check cooldown
        if not self._check_cooldown(symbol, bar.time):
            return None

        # 3. Session management
        today = bar.time.date().isoformat()
        if self._last_date != today:
            self._session_trades = {}
            self._traded_symbols_today = set()
            self._last_date = today

        # 4. No re-entry same symbol same day
        if symbol in self._traded_symbols_today:
            return None

        # 5. Max trades per session
        total_trades = sum(self._session_trades.values())
        if total_trades >= self.max_trades_per_session:
            return None

        # 6. Time window check
        current_time = time_utils.get_eastern_time_of_day(bar.time)
        if not (self.entry_start <= current_time <= self.entry_end):
            return None

        # 7. Require minimum bars for indicators
        if not self._require_min_bars(symbol_state, self.ema_slow + 10):
            return None

        bars = list(symbol_state.bars)

        # 8. Get EMAs for trend confirmation
        ema_fast, ema_slow = self._get_emas(bars, symbol_state)
        if ema_fast is None or ema_slow is None:
            return None

        is_uptrend = float(ema_fast) > float(ema_slow)
        is_downtrend = float(ema_fast) < float(ema_slow)

        # 9. Get breakout levels
        high_level, low_level = self._get_breakout_levels(bars, bar)
        if high_level is None or low_level is None:
            return None

        # 10. Check volume condition
        avg_vol = self._get_average_volume(symbol_state)
        if avg_vol is None or avg_vol <= 0:
            return None

        vol_ok = float(bar.volume) > (avg_vol * self.vol_mult)
        if not vol_ok:
            return None

        # 11. Check for breakout with strong close
        signal = None

        # BULLISH BREAKOUT
        if is_uptrend and market_state.regime == Regime.BULL:
            if bar.close > high_level and self._is_strong_close(bar, OrderSide.BUY):
                signal = self._create_long_signal(
                    symbol, bar, market_state, high_level, ema_fast, ema_slow, avg_vol
                )

        # BEARISH BREAKOUT
        if is_downtrend and market_state.regime == Regime.BEAR:
            if bar.close < low_level and self._is_strong_close(bar, OrderSide.SELL):
                signal = self._create_short_signal(
                    symbol, bar, market_state, low_level, ema_fast, ema_slow, avg_vol
                )

        if signal:
            self._traded_symbols_today.add(symbol)
            self._session_trades[symbol] = self._session_trades.get(symbol, 0) + 1

        return signal

    def _get_emas(
        self, bars: List[Bar], symbol_state: SymbolState
    ) -> tuple[Optional[float], Optional[float]]:
        """Get EMAs from cache or compute."""
        ema_fast = symbol_state.indicators.get(f"ema_close:{self.ema_fast}")
        ema_slow = symbol_state.indicators.get(f"ema_close:{self.ema_slow}")

        if ema_fast is not None and ema_slow is not None:
            return float(ema_fast), float(ema_slow)

        closes = [float(b.close) for b in bars]
        return (
            FeatureCalculator.calculate_ema(closes, self.ema_fast),
            FeatureCalculator.calculate_ema(closes, self.ema_slow),
        )

    def _get_breakout_levels(
        self, bars: List[Bar], current_bar: Bar
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Get N-day high and low levels for breakout detection.

        Uses prior days only (not including today).
        """
        today = current_bar.time.date()

        # Collect bars from prior days
        prior_days_bars = [b for b in bars if b.time.date() < today]

        if len(prior_days_bars) < self.breakout_lookback:
            return None, None

        # Get last N days worth of bars
        # Group by date
        days: Dict[str, List[Bar]] = {}
        for b in prior_days_bars:
            d = b.time.date().isoformat()
            if d not in days:
                days[d] = []
            days[d].append(b)

        # Get the last N days
        sorted_days = sorted(days.keys(), reverse=True)[: self.breakout_lookback]

        if len(sorted_days) < self.breakout_lookback:
            return None, None

        highs = []
        lows = []
        for d in sorted_days:
            day_bars = days[d]
            highs.append(max(b.high for b in day_bars))
            lows.append(min(b.low for b in day_bars))

        return max(highs), min(lows)

    def _get_average_volume(self, symbol_state: SymbolState) -> Optional[float]:
        """Get 20-bar average volume."""
        avg_vol = symbol_state.indicators.get("sma_vol:20")
        if avg_vol is not None:
            try:
                return float(avg_vol)
            except (TypeError, ValueError):
                pass

        bars = list(symbol_state.bars)
        vols = [float(b.volume) for b in bars[-20:]]
        if not vols:
            return None

        return sum(vols) / len(vols)

    def _is_strong_close(self, bar: Bar, side: OrderSide) -> bool:
        """Check if close is in favorable portion of bar range."""
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return False

        close_position = (bar.close - bar.low) / bar_range

        if side == OrderSide.BUY:
            return close_position >= self.close_position_threshold
        else:
            return close_position <= (1 - self.close_position_threshold)

    def _create_long_signal(
        self,
        symbol: str,
        bar: Bar,
        market_state: MarketState,
        breakout_level: float,
        ema_fast: float,
        ema_slow: float,
        avg_vol: float,
    ) -> Signal:
        """Create long breakout signal."""
        entry_price = bar.close

        # Stop at breakout bar low (or EMA20 if tighter)
        stop_price = max(bar.low, float(ema_fast) * 0.99)
        if stop_price >= entry_price:
            stop_price = entry_price * 0.99  # Fallback 1%

        # Target based on bar range
        bar_range = bar.high - bar.low
        target_price = entry_price + (bar_range * self.risk_reward)

        return self._create_signal(
            symbol=symbol,
            side=OrderSide.BUY,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={
                "trigger": "momentum_breakout_long",
                "breakout_level": breakout_level,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "vol_mult": round(float(bar.volume) / avg_vol, 2) if avg_vol else None,
            },
        )

    def _create_short_signal(
        self,
        symbol: str,
        bar: Bar,
        market_state: MarketState,
        breakdown_level: float,
        ema_fast: float,
        ema_slow: float,
        avg_vol: float,
    ) -> Signal:
        """Create short breakdown signal."""
        entry_price = bar.close

        # Stop at breakout bar high (or EMA20 if tighter)
        stop_price = min(bar.high, float(ema_fast) * 1.01)
        if stop_price <= entry_price:
            stop_price = entry_price * 1.01  # Fallback 1%

        # Target based on bar range
        bar_range = bar.high - bar.low
        target_price = entry_price - (bar_range * self.risk_reward)

        return self._create_signal(
            symbol=symbol,
            side=OrderSide.SELL,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={
                "trigger": "momentum_breakdown_short",
                "breakdown_level": breakdown_level,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "vol_mult": round(float(bar.volume) / avg_vol, 2) if avg_vol else None,
            },
        )
