"""
Centralized timezone and time window utilities.

This module eliminates duplicate timezone handling across strategies by providing
standardized functions for US equity market time conversions and validations.
"""

from datetime import datetime, time, timezone

import pytz  # type: ignore

from src.core.logger import StructuredLogger

# Cache timezone object for performance
_EASTERN_TZ = pytz.timezone("US/Eastern")
logger = StructuredLogger("TimeUtils")


def get_eastern_timezone() -> pytz.tzinfo.BaseTzInfo:
    """
    Get the US/Eastern timezone object.

    Returns:
        pytz timezone object for US/Eastern
    """
    return _EASTERN_TZ


def to_eastern_time(dt: datetime) -> datetime:
    """
    Convert any datetime to US/Eastern timezone.

    Args:
        dt: Datetime to convert (naive datetimes are assumed to be UTC)

    Returns:
        Datetime converted to US/Eastern timezone
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_EASTERN_TZ)


def get_eastern_time_of_day(dt: datetime) -> time:
    """
    Get the time-of-day in US/Eastern timezone.

    Args:
        dt: Datetime to convert (naive datetimes are assumed to be UTC)

    Returns:
        Time object representing the time-of-day in Eastern timezone
    """
    et_dt = to_eastern_time(dt)
    return et_dt.time()


def in_trading_window(dt: datetime, start: time, end: time, convert_to_eastern: bool = True) -> bool:
    """
    Check if a datetime falls within a trading time window.

    Args:
        dt: Datetime to check (naive datetimes are assumed to be UTC)
        start: Start time of the window (inclusive)
        end: End time of the window (inclusive)
        convert_to_eastern: If True, convert dt to Eastern time before checking

    Returns:
        True if the time-of-day falls within [start, end], False otherwise

    Example:
        >>> from datetime import time
        >>> dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)  # 9:30 AM ET
        >>> in_trading_window(dt, time(9, 30), time(16, 0))
        True
    """
    try:
        if convert_to_eastern:
            t = get_eastern_time_of_day(dt)
        else:
            t = dt.time()
        return start <= t <= end
    except Exception as exc:
        # Fail open - better to allow trading than halt on timezone errors
        logger.warning(
            "time_window_check_failed",
            error=str(exc),
            start=str(start),
            end=str(end),
            convert_to_eastern=convert_to_eastern,
            dt=str(dt),
            exc_info=True,
        )
        return True


def parse_time_string(time_str: str) -> time:
    """
    Parse a time string in HH:MM format.

    Args:
        time_str: Time string in "HH:MM" format (e.g., "09:30")

    Returns:
        time object

    Raises:
        ValueError: If time_str is not in valid format

    Example:
        >>> parse_time_string("09:30")
        datetime.time(9, 30)
    """
    raw = str(time_str).strip()
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM")

    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM") from exc

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time value: {time_str}. Hour must be 0-23 and minute 0-59")

    return time(hour, minute)


def in_time_window_str(dt: datetime, start_str: str, end_str: str, convert_to_eastern: bool = True) -> bool:
    """
    Convenience function to check time window using string times.

    Args:
        dt: Datetime to check
        start_str: Start time as "HH:MM" string
        end_str: End time as "HH:MM" string
        convert_to_eastern: If True, convert dt to Eastern time

    Returns:
        True if within window, False otherwise

    Example:
        >>> dt = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        >>> in_time_window_str(dt, "09:30", "16:00")
        True
    """
    try:
        start = parse_time_string(start_str)
        end = parse_time_string(end_str)
        return in_trading_window(dt, start, end, convert_to_eastern)
    except Exception as exc:
        logger.warning(
            "time_window_string_check_failed",
            error=str(exc),
            start_str=start_str,
            end_str=end_str,
            convert_to_eastern=convert_to_eastern,
            dt=str(dt),
            exc_info=True,
        )
        return True


# Standard US equity market hours (Eastern time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PREMARKET_START = time(4, 0)
AFTERHOURS_END = time(20, 0)


def is_regular_trading_hours(dt: datetime) -> bool:
    """
    Check if datetime is within regular trading hours (9:30 AM - 4:00 PM ET).

    Args:
        dt: Datetime to check

    Returns:
        True if within regular trading hours
    """
    return in_trading_window(dt, MARKET_OPEN, MARKET_CLOSE)


def is_extended_hours(dt: datetime) -> bool:
    """
    Check if datetime is within extended hours (pre-market or after-hours).

    Extended hours: 4:00 AM - 9:30 AM ET and 4:00 PM - 8:00 PM ET

    Args:
        dt: Datetime to check

    Returns:
        True if within extended trading hours (but not regular hours)
    """
    t = get_eastern_time_of_day(dt)

    # Pre-market: 4:00 AM - 9:30 AM
    in_premarket = PREMARKET_START <= t < MARKET_OPEN

    # After-hours: 4:00 PM - 8:00 PM
    in_afterhours = MARKET_CLOSE < t <= AFTERHOURS_END

    return in_premarket or in_afterhours
