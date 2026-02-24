from datetime import datetime, time
from unittest.mock import MagicMock

from src.core import time_utils


def test_to_eastern_time_logs_warning_for_naive_datetime(monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr(time_utils, "logger", mock_logger)

    time_utils.to_eastern_time(datetime(2024, 1, 15, 14, 30))

    mock_logger.warning.assert_called()


def test_in_trading_window_logs_error_on_exception(monkeypatch):
    class BadTime:
        def time(self):
            raise ValueError("boom")

    mock_logger = MagicMock()
    monkeypatch.setattr(time_utils, "logger", mock_logger)

    result = time_utils.in_trading_window(BadTime(), time(9, 30), time(16, 0), convert_to_eastern=False)

    assert result is True
    mock_logger.error.assert_called()
