"""Unit tests for MarketSession implementations."""

from datetime import datetime, timezone

import pytest
import pytz

from src.core.market_session import CryptoSession, USEquitySession

_EASTERN = pytz.timezone("US/Eastern")


class TestUSEquitySession:
    """Tests for standard US Equity session logic."""

    @pytest.fixture
    def session(self):
        return USEquitySession()

    def test_is_open_during_regular_hours(self, session):
        """Open 9:30-16:00 ET weekdays."""
        # Wed Jan 15 2024 10:00 ET (Open)
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=_EASTERN)
        assert session.is_open(dt) is True

    def test_is_closed_premarket(self, session):
        """Closed before 9:30 ET."""
        # Wed Jan 15 2024 9:29 ET (Closed)
        dt = datetime(2024, 1, 15, 9, 29, tzinfo=_EASTERN)
        assert session.is_open(dt) is False

    def test_is_closed_afterhours(self, session):
        """Closed at/after 16:00 ET."""
        # Wed Jan 15 2024 16:00 ET (Closed)
        dt = datetime(2024, 1, 15, 16, 0, tzinfo=_EASTERN)
        assert session.is_open(dt) is False

    def test_is_closed_weekends(self, session):
        """Closed on weekends."""
        # Sat Jan 20 2024 12:00 ET (Closed)
        dt = datetime(2024, 1, 20, 12, 0, tzinfo=_EASTERN)
        assert session.is_open(dt) is False

    def test_get_next_open_same_day(self, session):
        """Returns today at 9:30 if called before open."""
        # Wed Jan 15 2024 8:00 ET -> 9:30 ET same day
        dt = datetime(2024, 1, 15, 8, 0, tzinfo=_EASTERN)
        next_open = session.get_next_open(dt)
        assert next_open == dt.replace(hour=9, minute=30)
        assert next_open.date() == dt.date()

    def test_get_next_open_next_day(self, session):
        """Returns tomorrow at 9:30 if called after open."""
        # Wed Jan 15 2024 10:00 ET -> Thu Jan 16 9:30 ET
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=_EASTERN)
        next_open = session.get_next_open(dt)
        assert next_open == dt.replace(day=16, hour=9, minute=30)

    def test_get_next_open_over_weekend(self, session):
        """Returns Monday at 9:30 if called on Friday after close."""
        # Fri Jan 19 2024 17:00 ET -> Mon Jan 22 9:30 ET
        dt = datetime(2024, 1, 19, 17, 0, tzinfo=_EASTERN)
        next_open = session.get_next_open(dt)
        assert next_open == dt.replace(day=22, hour=9, minute=30)

    def test_should_flatten_near_close(self, session):
        """True if within force_flat_mins of close."""
        # Wed Jan 15 2024 15:50 ET (10 mins to close)
        dt = datetime(2024, 1, 15, 15, 50, tzinfo=_EASTERN)
        # Should flatten within 15 mins
        assert session.should_flatten(dt, force_flat_mins=15) is True

    def test_should_not_flatten_early(self, session):
        """False if far from close."""
        # Wed Jan 15 2024 15:00 ET (60 mins to close)
        dt = datetime(2024, 1, 15, 15, 0, tzinfo=_EASTERN)
        assert session.should_flatten(dt, force_flat_mins=15) is False


class TestCryptoSession:
    """Tests for Crypto session logic (24/7)."""

    @pytest.fixture
    def session(self):
        return CryptoSession()

    def test_always_open(self, session):
        """Open 24/7."""
        # Random times
        times = [
            datetime(2024, 1, 15, 3, 0, tzinfo=timezone.utc),  # 3 AM UTC
            datetime(2024, 1, 20, 12, 0, tzinfo=timezone.utc),  # Saturday noon
            datetime(2024, 1, 21, 23, 59, tzinfo=timezone.utc),  # Sunday midnight
        ]
        for t in times:
            assert session.is_open(t) is True

    def test_never_flattens(self, session):
        """Never triggers flattening."""
        dt = datetime(2024, 1, 15, 15, 59, tzinfo=timezone.utc)
        assert session.should_flatten(dt) is False

    def test_next_open_is_now(self, session):
        """Next open is effecitvely immediately."""
        dt = datetime.now(timezone.utc)
        assert session.get_next_open(dt) == dt
