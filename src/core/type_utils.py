"""Type conversion utilities with safe fallback handling."""

from typing import Any, Optional


def safe_float(value: Any) -> Optional[float]:
    """
    Convert value to float, returning None if conversion fails.

    Args:
        value: Any value to convert to float

    Returns:
        Float value or None if conversion fails

    Examples:
        >>> safe_float("123.45")
        123.45
        >>> safe_float("invalid")
        None
        >>> safe_float(None)
        None
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    """
    Convert value to int, returning None if conversion fails.

    Args:
        value: Any value to convert to int

    Returns:
        Integer value or None if conversion fails

    Examples:
        >>> safe_int("123")
        123
        >>> safe_int("123.9")
        123
        >>> safe_int("invalid")
        None
        >>> safe_int(None)
        None
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
