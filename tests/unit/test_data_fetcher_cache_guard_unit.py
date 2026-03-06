from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.fetcher import DataFetcher


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_bars_handles_naive_cached_timestamp_without_crashing() -> None:
    logger = MagicMock()
    settings = SimpleNamespace(
        use_gateway_data=False,
        use_heber_storage=False,
        cerberus_failover_to_legacy=True,
        cerberus_data_backend="legacy",
        cerberus_heber_data_root="",
    )

    with patch("src.data.fetcher.get_settings", return_value=settings):
        fetcher = DataFetcher(
            alpaca_client=MagicMock(),
            unusual_whales_client=MagicMock(),
            logger=logger,
        )

    start = datetime(2025, 1, 10, 14, 30, tzinfo=timezone.utc)
    end = datetime(2025, 1, 10, 14, 35, tzinfo=timezone.utc)
    fetcher._bars_cache["AAPL"] = {
        "start": start,
        "bars": [
            {
                "t": "2025-01-10T14:32:00",
                "o": 100,
                "h": 101,
                "l": 99,
                "c": 100.5,
                "v": 1000,
            }
        ],
    }
    fetcher._fetch_alpaca_bars_internal = AsyncMock(return_value=[])

    bars, metrics = await fetcher.fetch_bars("AAPL", start, end, timeframe="1Min")

    assert bars
    assert metrics["cache_hits"] == 1
    logger.warning.assert_any_call(
        "Naive cached bar timestamp assumed UTC",
        event_type="data_validation",
        symbol="AAPL",
        raw_timestamp="2025-01-10T14:32:00",
    )
