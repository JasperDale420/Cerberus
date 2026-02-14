"""Live bar buffer for accumulating WebSocket bars.

Captures real-time bars from the WebSocket stream and makes them available
to the scanner's DataFetcher, bypassing 15-min delayed REST endpoints.
"""

from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class LiveBarBuffer:
    """In-memory ring buffer for live WebSocket bars.

    Stores up to `max_bars_per_symbol` bars per symbol in a deque.
    Bars are stored as dicts with a timestamp for range queries.
    """

    def __init__(self, max_bars_per_symbol: int = 1500, max_symbols: int = 50):
        self._buffers: OrderedDict[str, deque] = OrderedDict()
        self._max_bars = max_bars_per_symbol
        self._max_symbols = max_symbols

    def append(self, symbol: str, bar: Dict[str, Any]) -> None:
        """Append a bar to the buffer for a symbol.

        The bar dict should contain a timestamp field ('t' or 'timestamp').
        """
        sym = symbol.upper()
        if sym not in self._buffers:
            self._buffers[sym] = deque(maxlen=self._max_bars)
            # LRU eviction
            while len(self._buffers) > self._max_symbols:
                self._buffers.popitem(last=False)

        self._buffers[sym].append(bar)
        self._buffers.move_to_end(sym)

    def seed(self, symbol: str, bars: List[Dict[str, Any]]) -> None:
        """Seed the buffer with historical bars (e.g. from initial REST fetch).

        Replaces any existing data for the symbol.
        """
        sym = symbol.upper()
        buf = deque(maxlen=self._max_bars)
        buf.extend(bars)
        self._buffers[sym] = buf
        self._buffers.move_to_end(sym)

        while len(self._buffers) > self._max_symbols:
            self._buffers.popitem(last=False)

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> Optional[List[Dict[str, Any]]]:
        """Return bars within [start, end] for a symbol.

        Returns None if the symbol has no buffered data.
        Returns an empty list if data exists but none falls in the window.
        """
        sym = symbol.upper()
        buf = self._buffers.get(sym)
        if buf is None or len(buf) == 0:
            return None

        result = []
        for bar in buf:
            bar_time = self._parse_time(bar)
            if bar_time is None:
                continue
            if start <= bar_time <= end:
                result.append(bar)

        return result

    def has_data(self, symbol: str) -> bool:
        """Check if the buffer has any data for a symbol."""
        sym = symbol.upper()
        buf = self._buffers.get(sym)
        return buf is not None and len(buf) > 0

    def bar_count(self, symbol: str) -> int:
        """Return the number of buffered bars for a symbol."""
        sym = symbol.upper()
        buf = self._buffers.get(sym)
        return len(buf) if buf else 0

    @staticmethod
    def _parse_time(bar: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from a bar dict."""
        raw = bar.get("t") or bar.get("timestamp")
        if raw is None:
            return None
        if isinstance(raw, datetime):
            if raw.tzinfo is None:
                return raw.replace(tzinfo=timezone.utc)
            return raw
        # String timestamp
        try:
            ts = str(raw)
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None
