from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.core.logger import StructuredLogger
from src.data.fetcher import DataFetcher


@pytest.mark.unit
def test_resolve_fetch_start_accepts_datetime_timestamp() -> None:
    fetcher = DataFetcher(
        alpaca_client=MagicMock(),
        unusual_whales_client=MagicMock(),
        logger=StructuredLogger("test_fetcher_cache_ts", level="INFO"),
    )

    start = datetime(2025, 1, 10, tzinfo=timezone.utc)
    last_ts = datetime(2025, 1, 10, 14, 30, tzinfo=timezone.utc)
    symbol = "AAPL"

    fetcher._bars_cache[symbol] = {"start": start, "bars": [{"t": last_ts}]}

    resolved_start, existing = fetcher._resolve_fetch_start(symbol, start)

    assert existing == [{"t": last_ts}]
    assert resolved_start == last_ts + timedelta(seconds=1)
