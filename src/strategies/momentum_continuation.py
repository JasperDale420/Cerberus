"""
Momentum Continuation Strategy.

Rides initial breakout momentum without waiting for pullbacks.
Captures the first leg of breakouts that pullback strategies miss.
"""

from __future__ import annotations

from datetime import time as time_type
from typing import Any, Dict, List, Optional

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
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
    DEFAULT_BREAKOUT_LOOKBACK = 3  # Days for high/low
    DEFAULT_VOL_MULT = 1.2  # Volume > 1.2x average
    DEFAULT_CLOSE_POSITION = 0.6  # Close in top 40% of range
    DEFAULT_RISK_REWARD = 2.0
    DEFAULT_STOP_PCT = 0.005  # 0.5% stop loss from entry

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)

        # Breakout parameters
        self.breakout_lookback = int(config.get("breakout_lookback", self.DEFAULT_BREAKOUT_LOOKBACK))
        self.vol_mult = float(config.get("vol_mult", self.DEFAULT_VOL_MULT))
        self.close_position_threshold = float(
            config.get("close_position_threshold", config.get("close_position", self.DEFAULT_CLOSE_POSITION))
        )
        self.risk_reward = float(config.get("risk_reward", self.DEFAULT_RISK_REWARD))
        self.stop_pct = float(config.get("stop_pct", self.DEFAULT_STOP_PCT))

        # EMA for trend confirmation
        self.ema_fast = int(config.get("ema_fast", 20))
        self.ema_slow = int(config.get("ema_slow", 50))

        # Time window (morning momentum)
        self.entry_start = time_type(9, 35)
        self.entry_end = time_type(15, 0)

        # Position management
        self.max_trades_per_session = int(config.get("max_trades_per_session", 2))
        self.slippage_allowance = float(config.get("slippage_allowance", 0.001))  # 10 bps default leeway

        self._session_trades: Dict[str, int] = {}
        self._traded_symbols_today: set = set()
        self._last_date: Optional[str] = None
        self._entries_per_timestamp: Dict[str, int] = {}  # Limit correlated entries
        self._last_timestamp: Optional[str] = None

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # 1. Regime gating
        # Breakouts require strong conditions. Bypass if market_state does not match our activation config.
        if market_state.regime in ["chop", "risk_off"]:
            if len(symbol_state.bars) % 200 == 0:
                self.logger.warning(f"MomentumCont: Rejected by regime {market_state.regime}")
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

        # Require meaningful EMA spread (0.1% minimum separation)
        ema_spread = abs(float(ema_fast) - float(ema_slow)) / float(ema_slow)
        if ema_spread < 0.001:
            return None

        # 9. Get breakout levels
        high_level, low_level = self._get_breakout_levels(symbol_state)
        if high_level is None or low_level is None:
            # Debug: Log when breakout levels unavailable
            if len(bars) % 100 == 0:
                self.logger.warning(
                    "MomentumCont: No breakout levels",
                    symbol=symbol,
                    bar_count=len(bars),
                    lookback_days=self.breakout_lookback,
                )
            return None

        # 10. Check volume condition
        avg_vol = self._get_average_volume(symbol_state)
        if avg_vol is None or avg_vol <= 0:
            return None

        vol_ok = float(bar.volume) > (avg_vol * self.vol_mult)
        if not vol_ok:
            return None

        # Debug: Log periodically when no breakout condition met
        if len(bars) % 200 == 0:
            self.logger.warning(
                "MomentumCont: No breakout",
                symbol=symbol,
                close=round(bar.close, 2),
                high_level=round(high_level, 2),
                low_level=round(low_level, 2),
                uptrend=is_uptrend,
                downtrend=is_downtrend,
            )

        # 11. Limit correlated entries (max 2 per timestamp to avoid cluster losses)
        ts_key = bar.time.isoformat()[:16]  # Minute-level grouping
        if ts_key != self._last_timestamp:
            self._entries_per_timestamp = {}
            self._last_timestamp = ts_key
        if self._entries_per_timestamp.get(ts_key, 0) >= 2:
            return None

        # 12. Check for breakout with strong close and margin
        signal = None
        breakout_margin = 0.001  # Require 0.1% beyond breakout level

        # BULLISH BREAKOUT
        if is_uptrend:
            threshold = high_level * (1.0 + breakout_margin)
            if bar.close > threshold and self._is_strong_close(bar, OrderSide.BUY):
                signal = self._create_long_signal(
                    symbol, bar, symbol_state, market_state, high_level, ema_fast, ema_slow, avg_vol
                )

        # BEARISH BREAKOUT
        if is_downtrend:
            threshold = low_level * (1.0 - breakout_margin)
            if bar.close < threshold and self._is_strong_close(bar, OrderSide.SELL):
                signal = self._create_short_signal(
                    symbol, bar, symbol_state, market_state, low_level, ema_fast, ema_slow, avg_vol
                )

        if signal:
            self._traded_symbols_today.add(symbol)
            self._session_trades[symbol] = self._session_trades.get(symbol, 0) + 1
            self._entries_per_timestamp[ts_key] = self._entries_per_timestamp.get(ts_key, 0) + 1

        return signal

    def _get_emas(self, bars: List[Bar], symbol_state: SymbolState) -> tuple[Optional[float], Optional[float]]:
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

    def _get_breakout_levels(self, symbol_state: SymbolState) -> tuple[Optional[float], Optional[float]]:
        """
        Get high and low levels for breakout detection.
        Uses prior day levels from SymbolState.meta or multi-day levels from indicators.
        """
        daily_highs = symbol_state.indicators.get("daily_highs")
        daily_lows = symbol_state.indicators.get("daily_lows")

        if daily_highs and daily_lows:
            lookback = min(len(daily_highs), self.breakout_lookback)
            lookback_highs = list(daily_highs)[-lookback:]
            lookback_lows = list(daily_lows)[-lookback:]
            prior_high = max(lookback_highs)
            prior_low = min(lookback_lows)
        else:
            prior_high = symbol_state.meta.get("prior_day_high")
            prior_low = symbol_state.meta.get("prior_day_low")

        if prior_high is None or prior_low is None:
            return None, None

        return float(prior_high), float(prior_low)

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
        symbol_state: SymbolState,
        market_state: MarketState,
        breakout_level: float,
        ema_fast: float,
        ema_slow: float,
        avg_vol: float,
    ) -> Signal:
        """Create long breakout signal."""
        entry_price = bar.close * (1.0 + self.slippage_allowance)

        # Percentage-based stop loss — guarantees meaningful distance from entry
        stop_price = entry_price * (1.0 - self.stop_pct)

        # Widen with ATR if available
        atr = float(symbol_state.meta.get("atr", 0.0) or 0.0)
        if atr > 0:
            atr_stop = entry_price - (atr * 1.5)
            stop_price = min(stop_price, atr_stop)  # Use wider of the two

        # Risk/reward target
        risk = abs(entry_price - stop_price)
        target_price = entry_price + (risk * self.risk_reward)

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
        symbol_state: SymbolState,
        market_state: MarketState,
        breakdown_level: float,
        ema_fast: float,
        ema_slow: float,
        avg_vol: float,
    ) -> Signal:
        """Create short breakdown signal."""
        entry_price = bar.close * (1.0 - self.slippage_allowance)

        # Percentage-based stop loss — guarantees meaningful distance from entry
        stop_price = entry_price * (1.0 + self.stop_pct)

        # Widen with ATR if available
        atr = float(symbol_state.meta.get("atr", 0.0) or 0.0)
        if atr > 0:
            atr_stop = entry_price + (atr * 1.5)
            stop_price = max(stop_price, atr_stop)  # Use wider of the two

        # Risk/reward target
        risk = abs(entry_price - stop_price)
        target_price = entry_price - (risk * self.risk_reward)

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
