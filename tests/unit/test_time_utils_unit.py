from datetime import time

import pytest

from src.core.time_utils import parse_time_string


@pytest.mark.unit
def test_parse_time_string_valid() -> None:
    assert parse_time_string("09:30") == time(9, 30)
    assert parse_time_string("9:05") == time(9, 5)


@pytest.mark.unit
def test_parse_time_string_invalid_format_message() -> None:
    with pytest.raises(ValueError, match=r"Expected HH:MM"):
        parse_time_string("0930")


@pytest.mark.unit
def test_parse_time_string_invalid_numeric_message() -> None:
    with pytest.raises(ValueError, match=r"Expected HH:MM"):
        parse_time_string("09:xx")


@pytest.mark.unit
def test_parse_time_string_out_of_range_message() -> None:
    with pytest.raises(ValueError, match=r"Hour must be 0-23"):
        parse_time_string("25:00")
