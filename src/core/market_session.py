"""
Market session abstraction to support multi-asset trading (e.g., US Equities vs Crypto).
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from src.core.time_utils import (
    _EASTERN_TZ,
    MARKET_CLOSE,
    MARKET_OPEN,
)


class MarketSession(ABC):
    """Abstract base class for market session logic."""

    @abstractmethod
    def is_open(self, dt: datetime) -> bool:
        """Return True if the market is open for trading at the given datetime."""
        pass

    @abstractmethod
    def get_next_open(self, dt: datetime) -> datetime:
        """Return the next market open datetime relative to the given datetime."""
        pass

    @abstractmethod
    def should_flatten(self, dt: datetime, force_flat_mins: int = 15) -> bool:
        """
        Return True if positions should be flattened due to approaching market close.

        Args:
            dt: Current datetime.
            force_flat_mins: Minutes before close to trigger flattening.
        """
        pass

    @abstractmethod
    def get_close_time(self, dt: datetime) -> datetime | None:
        """Return the market close time for the current/next session, or None if 24/7."""
        pass


class USEquitySession(MarketSession):
    """
    Standard US Equity market session.
    Open: 9:30 AM - 4:00 PM ET, Monday-Friday.
    Closed: Weekends and Holidays (holiday logic TBD/simplified for now).
    """

    def is_open(self, dt: datetime) -> bool:
        """
        Return True when within regular US session hours on a weekday.
        Session window is 09:30 <= time < 16:00 ET.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        et_time = dt.astimezone(_EASTERN_TZ)

        if et_time.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        t = et_time.time()
        if t < MARKET_OPEN or t >= MARKET_CLOSE:
            return False

        return True

    def get_next_open(self, dt: datetime) -> datetime:
        """
        Return the next regular US market open (09:30 ET on a weekday).
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        et_time = dt.astimezone(_EASTERN_TZ)
        today_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)

        if et_time < today_open and et_time.weekday() < 5:
            return today_open

        # Start checking from tomorrow
        candidate = today_open + timedelta(days=1)
        while candidate.weekday() >= 5:  # Skip weekends
            candidate += timedelta(days=1)
        return candidate

    def get_close_time(self, dt: datetime) -> datetime | None:
        """Return today's close (16:00 ET) if weekday, else None (no session today)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        et_time = dt.astimezone(_EASTERN_TZ)
        if et_time.weekday() >= 5:
            return None

        return et_time.replace(hour=16, minute=0, second=0, microsecond=0)

    def should_flatten(self, dt: datetime, force_flat_mins: int = 15) -> bool:
        """Return True if within `force_flat_mins` of 16:00 ET close."""
        if not self.is_open(dt):
            return False  # Already closed or pre-market

        close_time = self.get_close_time(dt)
        if close_time is None:
            return False

        # Calculate time remaining until close
        time_to_close = (close_time - dt).total_seconds() / 60.0
        return 0 < time_to_close <= force_flat_mins


class CryptoSession(MarketSession):
    """
    Crypto market session.
    Open: 24/7/365.
    Never closes, never flattens (unless manual intervention).
    """

    def is_open(self, dt: datetime) -> bool:
        return True

    def get_next_open(self, dt: datetime) -> datetime:
        # It's always open, so effectively "now"
        return dt

    def get_close_time(self, dt: datetime) -> datetime | None:
        return None

    def should_flatten(self, dt: datetime, force_flat_mins: int = 15) -> bool:
        return False
