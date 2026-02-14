from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.core.logger import StructuredLogger
from src.data.fetcher import DataFetcher


def _logger(name: str) -> StructuredLogger:
    return StructuredLogger(name, level="INFO")


@pytest.mark.unit
def test_resolve_fetch_start_handles_object_bar_timestamps() -> None:
    class _Bar:
        def __init__(self, t: str) -> None:
            self.t = t

    alpaca = MagicMock()
    uw = MagicMock()
    fetcher = DataFetcher(alpaca, uw, _logger("test_fetcher_cache"))

    start = datetime(2026, 2, 14, 14, 0, tzinfo=timezone.utc)
    last_ts = datetime(2026, 2, 14, 14, 30, tzinfo=timezone.utc).isoformat()
    fetcher._bars_cache["AAPL"] = {"start": start, "bars": [_Bar(last_ts)]}

    resolved_start, existing = fetcher._resolve_fetch_start("AAPL", start)

    assert resolved_start == datetime(2026, 2, 14, 14, 30, 1, tzinfo=timezone.utc)
    assert existing == fetcher._bars_cache["AAPL"]["bars"]
