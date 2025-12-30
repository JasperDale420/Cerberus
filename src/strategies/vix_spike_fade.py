"""
VIX Spike Fade Strategy.

Fades extreme VIX spikes by buying SPY/QQQ when fear is overextended.
This strategy specifically activates during SHOCK volatility regimes
when all other strategies are disabled.
"""

from __future__ import annotations

from datetime import time as time_type
from typing import Any, Dict, Optional

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class VixSpikeFadeStrategy(BaseStrategy):
    """
    VIX Spike Fade Strategy.

    Fades extreme fear by buying SPY/QQQ when VIX spikes > 20% intraday.
    Targets partial reversion of the day's decline.

    Activation: vol=shock, risk=risk_off (counter-trade fear)
    """

    name = "vix_spike_fade"

    # Default thresholds
    DEFAULT_VIX_SPIKE_PCT = 0.20  # 20% intraday VIX spike
    DEFAULT_VIX_ABSOLUTE = 30.0  # Or absolute VIX > 30
    DEFAULT_INDEX_DROP_PCT = 0.015  # 1.5% down from open
    DEFAULT_REVERSION_TARGET = 0.50  # Target 50% of decline
    DEFAULT_STOP_BUFFER = 0.005  # 0.5% below day low

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)

        # Allowed symbols (index ETFs only)
        self.allowed_symbols = {
            s.upper() for s in config.get("symbols", ["SPY", "QQQ"])
        }

        # VIX thresholds
        self.vix_spike_pct = float(
            config.get("vix_spike_pct", self.DEFAULT_VIX_SPIKE_PCT)
        )
        self.vix_absolute = float(config.get("vix_absolute", self.DEFAULT_VIX_ABSOLUTE))

        # Index drop threshold
        self.index_drop_pct = float(
            config.get("index_drop_pct", self.DEFAULT_INDEX_DROP_PCT)
        )

        # Exit parameters
        self.reversion_target = float(
            config.get("reversion_target", self.DEFAULT_REVERSION_TARGET)
        )
        self.stop_buffer = float(config.get("stop_buffer", self.DEFAULT_STOP_BUFFER))

        # Time window (after dust settles)
        self.entry_start = time_type(10, 30)  # After 10:30 ET
        self.entry_end = time_type(15, 30)  # Before 15:30 ET

        # Track if we've traded this shock event
        self._traded_today = False
        self._last_trade_date: Optional[str] = None

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # 1. Symbol filter (SPY/QQQ only)
        if str(symbol).upper() not in self.allowed_symbols:
            return None

        # 2. Check cooldown
        if not self._check_cooldown(symbol, bar.time):
            return None

        # 3. One trade per shock event
        today = bar.time.date().isoformat()
        if self._last_trade_date == today and self._traded_today:
            return None
        if self._last_trade_date != today:
            self._traded_today = False
            self._last_trade_date = today

        # 4. Time window check
        current_time = time_utils.get_eastern_time_of_day(bar.time)
        if not (self.entry_start <= current_time <= self.entry_end):
            return None

        # 5. Get VIX data from market_state meta
        vix_data = self._get_vix_data(market_state)
        if vix_data is None:
            return None

        vix_current, vix_open, vix_change_pct = vix_data

        # 6. Check VIX spike conditions
        is_vix_spike = (
            vix_change_pct >= self.vix_spike_pct or vix_current >= self.vix_absolute
        )
        if not is_vix_spike:
            return None

        # 7. Check index drop condition
        open_price = self._get_session_open(symbol_state)
        if open_price is None or open_price <= 0:
            return None

        decline_pct = (open_price - bar.close) / open_price
        if decline_pct < self.index_drop_pct:
            return None

        # 8. Check for reversal candle (hammer/doji-like)
        if not self._is_reversal_candle(bar):
            return None

        # 9. Calculate entry, stop, target
        entry_price = bar.close
        day_low = self._get_session_low(symbol_state, bar)
        stop_price = day_low * (1 - self.stop_buffer)

        # Target: partial reversion toward open
        decline_amount = open_price - entry_price
        target_price = entry_price + (decline_amount * self.reversion_target)

        # Sanity checks
        if stop_price >= entry_price:
            stop_price = entry_price * 0.99  # Fallback 1% stop

        if target_price <= entry_price:
            return None  # No upside

        # 10. Create signal (LONG only - fading fear)
        self._traded_today = True

        return self._create_signal(
            symbol=symbol,
            side=OrderSide.BUY,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={
                "trigger": "vix_spike_fade",
                "vix_current": vix_current,
                "vix_open": vix_open,
                "vix_change_pct": round(vix_change_pct, 4),
                "index_decline_pct": round(decline_pct, 4),
                "session_open": open_price,
                "day_low": day_low,
            },
        )

    def _get_vix_data(
        self, market_state: MarketState
    ) -> Optional[tuple[float, float, float]]:
        """
        Extract VIX data from market_state.meta.

        Returns (current, open, change_pct) or None if unavailable.
        """
        meta = getattr(market_state, "meta", None)
        if not isinstance(meta, dict):
            return None

        vix_current = meta.get("vix_current") or meta.get("vix")
        vix_open = meta.get("vix_open") or meta.get("vix_session_open")

        if vix_current is None:
            return None

        try:
            vix_current_f = float(vix_current)
            vix_open_f = float(vix_open) if vix_open else vix_current_f
        except (TypeError, ValueError):
            return None

        if vix_open_f <= 0:
            return None

        change_pct = (vix_current_f - vix_open_f) / vix_open_f

        return vix_current_f, vix_open_f, change_pct

    def _get_session_open(self, symbol_state: SymbolState) -> Optional[float]:
        """Get session open price from first bar or cache."""
        cached = symbol_state.indicators.get("session_open")
        if cached is not None:
            try:
                return float(cached)
            except (TypeError, ValueError):
                pass

        # Find first bar of session
        bars = list(symbol_state.bars)
        if not bars:
            return None

        today = bars[-1].time.date()
        for b in bars:
            if b.time.date() == today:
                return float(b.open)

        return None

    def _get_session_low(self, symbol_state: SymbolState, current_bar: Bar) -> float:
        """Get session low from bars."""
        bars = list(symbol_state.bars)
        today = current_bar.time.date()

        lows = [b.low for b in bars if b.time.date() == today]
        lows.append(current_bar.low)

        return min(lows) if lows else current_bar.low

    def _is_reversal_candle(self, bar: Bar) -> bool:
        """
        Check if bar shows reversal characteristics (hammer, doji, etc.).

        Criteria:
        - Close in upper 40% of bar range
        - Lower wick at least 1.5x body size
        """
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return False

        body = abs(bar.close - bar.open)
        close_position = (bar.close - bar.low) / bar_range

        # Close in upper portion of range
        if close_position < 0.6:
            return False

        # Lower wick should be significant
        lower_wick = min(bar.open, bar.close) - bar.low
        if body > 0 and lower_wick < (body * 1.5):
            return False

        return True
