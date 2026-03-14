"""Streaming Scanner — incremental microstructure feature updates between full scans.

Uses live WebSocket quote and trade data to keep OFI, TFI, and high-low spread
fresh in real-time, hot-patching SymbolState metadata between periodic full scans.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from src.core.logger import StructuredLogger
from src.data.calculator import FeatureCalculator
from src.data.client import StreamQuote, StreamTrade

# Minimum threshold for feature change to trigger an update callback
OFI_CHANGE_THRESHOLD = 0.15
TFI_CHANGE_THRESHOLD = 0.15

# Default rolling window duration
DEFAULT_WINDOW_MINUTES = 5

# Minimum quotes/trades needed for meaningful calculation
MIN_QUOTES_FOR_OFI = 5
MIN_TRADES_FOR_TFI = 10


@dataclass
class StreamingFeatureState:
    """Per-symbol rolling state for streaming microstructure calculations."""

    symbol: str
    recent_quotes: deque = field(default_factory=lambda: deque(maxlen=2000))
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=5000))
    last_ofi: float = 0.0
    last_tfi: float = 0.0
    last_high_low_spread: float = 0.0
    last_update: Optional[datetime] = None
    # Track high/low from trade prices for spread calculation
    session_high: float = 0.0
    session_low: float = float("inf")


class StreamingScanner:
    """Incrementally updates microstructure features from live quote/trade data.

    Maintains a rolling window of recent quotes and trades per symbol.
    On each new message, recalculates OFI or TFI and emits an update
    callback when the feature changes significantly.
    """

    def __init__(
        self,
        calculator: FeatureCalculator,
        logger: StructuredLogger,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
        on_update: Optional[Callable[[str, Dict[str, float]], Any]] = None,
    ):
        self._calculator = calculator
        self._logger = logger
        self._window = timedelta(minutes=window_minutes)
        self._on_update = on_update
        self._states: Dict[str, StreamingFeatureState] = {}
        self._lock = asyncio.Lock()

    def _get_state(self, symbol: str) -> StreamingFeatureState:
        if symbol not in self._states:
            self._states[symbol] = StreamingFeatureState(symbol=symbol)
        return self._states[symbol]

    def _evict_stale(self, window: deque, cutoff: datetime) -> None:
        """Remove entries older than the cutoff from the front of the deque."""
        while window and window[0].get("_ts", datetime.min.replace(tzinfo=timezone.utc)) < cutoff:
            window.popleft()

    def add_symbol(self, symbol: str) -> None:
        """Register a symbol for streaming tracking."""
        self._get_state(symbol)

    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from streaming tracking."""
        self._states.pop(symbol, None)

    @property
    def tracked_symbols(self) -> set[str]:
        return set(self._states.keys())

    async def on_quote(self, symbol: str, quote: StreamQuote) -> None:
        """Process an incoming quote and update OFI if threshold exceeded."""
        state = self._get_state(symbol)
        now = quote.timestamp

        # Store as dict matching calculator.calculate_ofi expectations
        entry = {
            "bp": quote.bid_price,
            "bs": quote.bid_size,
            "ap": quote.ask_price,
            "as": quote.ask_size,
            "_ts": now,
        }
        state.recent_quotes.append(entry)

        # Evict stale quotes
        cutoff = now - self._window
        self._evict_stale(state.recent_quotes, cutoff)

        # Recalculate OFI
        if len(state.recent_quotes) >= MIN_QUOTES_FOR_OFI:
            new_ofi = self._calculator.calculate_ofi(list(state.recent_quotes))
            delta = abs(new_ofi - state.last_ofi)

            if delta >= OFI_CHANGE_THRESHOLD:
                state.last_ofi = new_ofi
                state.last_update = now
                await self._emit_update(symbol, state)

    async def on_trade(self, symbol: str, trade: StreamTrade) -> None:
        """Process an incoming trade and update TFI if threshold exceeded."""
        state = self._get_state(symbol)
        now = trade.timestamp

        # Store as dict matching calculator.calculate_tfi expectations
        entry = {
            "p": trade.price,
            "s": trade.size,
            "_ts": now,
        }
        state.recent_trades.append(entry)

        # Track session high/low for spread calculation
        if trade.price > state.session_high:
            state.session_high = trade.price
        if trade.price < state.session_low:
            state.session_low = trade.price

        # Evict stale trades
        cutoff = now - self._window
        self._evict_stale(state.recent_trades, cutoff)

        # Recalculate TFI
        if len(state.recent_trades) >= MIN_TRADES_FOR_TFI:
            new_tfi = self._calculator.calculate_tfi(list(state.recent_trades))
            delta = abs(new_tfi - state.last_tfi)

            if delta >= TFI_CHANGE_THRESHOLD:
                state.last_tfi = new_tfi
                state.last_update = now

                # Update high-low spread from trades
                if state.session_low < float("inf") and state.session_high > 0:
                    mid = (state.session_high + state.session_low) / 2
                    if mid > 0:
                        state.last_high_low_spread = (state.session_high - state.session_low) / mid

                await self._emit_update(symbol, state)

    async def _emit_update(self, symbol: str, state: StreamingFeatureState) -> None:
        """Emit a feature update to the registered callback."""
        if self._on_update is None:
            return

        update = {
            "ofi": state.last_ofi,
            "tfi": state.last_tfi,
            "high_low_spread": state.last_high_low_spread,
        }

        try:
            if asyncio.iscoroutinefunction(self._on_update):
                await self._on_update(symbol, update)
            else:
                self._on_update(symbol, update)
        except Exception as e:
            self._logger.warning(
                "Streaming scanner update callback failed",
                symbol=symbol,
                error=str(e),
            )

    def get_features(self, symbol: str) -> Dict[str, float]:
        """Get the latest streaming features for a symbol."""
        state = self._states.get(symbol)
        if state is None:
            return {"ofi": 0.0, "tfi": 0.0, "high_low_spread": 0.0}
        return {
            "ofi": state.last_ofi,
            "tfi": state.last_tfi,
            "high_low_spread": state.last_high_low_spread,
        }

    def reset_session(self) -> None:
        """Reset all session-level state (e.g., at market open)."""
        for state in self._states.values():
            state.recent_quotes.clear()
            state.recent_trades.clear()
            state.last_ofi = 0.0
            state.last_tfi = 0.0
            state.last_high_low_spread = 0.0
            state.session_high = 0.0
            state.session_low = float("inf")
            state.last_update = None
