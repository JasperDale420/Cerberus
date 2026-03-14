from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.data.client import UnifiedDataClient
from src.data.fetcher import DataFetcher


@pytest.fixture
def mock_client():
    client = MagicMock(spec=UnifiedDataClient)
    client.get_historical_bars = MagicMock(
        return_value={
            "bars": [{"t": "2026-03-13T10:00:00Z", "o": 150.0, "h": 151.0, "l": 149.0, "c": 150.5, "v": 1000}]
        }
    )
    client.get_trades = MagicMock(return_value=[{"t": "2026-03-13T10:00:00Z", "p": 150.0, "s": 100}])
    client.get_quotes = MagicMock(return_value=[{"t": "2026-03-13T10:00:00Z", "bp": 150.0, "ap": 150.1}])
    client.get_flow = MagicMock(return_value={"data": [{"flow": "test"}]})
    client.get_gex = MagicMock(return_value=[{"gex": 1.0}])
    client.get_prior_day_stats = MagicMock(return_value=(151.0, 149.0, 150.5))
    client.get_avg_daily_volume = MagicMock(return_value=1_000_000.0)
    return client


@pytest.fixture
def mock_uw():
    return MagicMock()


@pytest.fixture
def mock_logger():
    return MagicMock()


def test_fetcher_uses_unified_client(mock_client, mock_uw, mock_logger):
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    assert fetcher.unified_client is mock_client


def test_fetcher_no_alpaca_or_central_api_params(mock_client, mock_uw, mock_logger):
    """Verify old params are gone."""
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    assert not hasattr(fetcher, "alpaca_client")
    assert not hasattr(fetcher, "central_api_client")
    assert not hasattr(fetcher, "heber_client")
    assert not hasattr(fetcher, "data_backend")


def test_fetch_prior_day_stats(mock_client, mock_uw, mock_logger):
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    high, low, close = fetcher.fetch_prior_day_stats("AAPL", datetime.now(timezone.utc))
    assert high == 151.0
    assert close == 150.5


def test_fetch_prior_day_stats_error_returns_zeros(mock_client, mock_uw, mock_logger):
    mock_client.get_prior_day_stats.side_effect = Exception("network error")
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    high, low, close = fetcher.fetch_prior_day_stats("AAPL", datetime.now(timezone.utc))
    assert (high, low, close) == (0.0, 0.0, 0.0)


def test_fetch_avg_daily_volume(mock_client, mock_uw, mock_logger):
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    vol = fetcher.fetch_avg_daily_volume("AAPL", datetime.now(timezone.utc), 20)
    assert vol == 1_000_000.0


def test_fetch_avg_daily_volume_zero_lookback(mock_client, mock_uw, mock_logger):
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    vol = fetcher.fetch_avg_daily_volume("AAPL", datetime.now(timezone.utc), 0)
    assert vol is None
    mock_client.get_avg_daily_volume.assert_not_called()


def test_fetch_avg_daily_volume_error_returns_none(mock_client, mock_uw, mock_logger):
    mock_client.get_avg_daily_volume.side_effect = Exception("timeout")
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    vol = fetcher.fetch_avg_daily_volume("AAPL", datetime.now(timezone.utc), 20)
    assert vol is None


@pytest.mark.asyncio
async def test_fetch_flow(mock_client, mock_uw, mock_logger):
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    result = await fetcher.fetch_flow("AAPL", "2026-03-13")
    assert result == [{"flow": "test"}]


@pytest.mark.asyncio
async def test_fetch_flow_error_returns_empty(mock_client, mock_uw, mock_logger):
    mock_client.get_flow.side_effect = Exception("fail")
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    result = await fetcher.fetch_flow("AAPL", "2026-03-13")
    assert result == []


@pytest.mark.asyncio
async def test_fetch_gex(mock_client, mock_uw, mock_logger):
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    result = await fetcher.fetch_gex("AAPL")
    assert result == [{"gex": 1.0}]


@pytest.mark.asyncio
async def test_fetch_gex_error_returns_empty(mock_client, mock_uw, mock_logger):
    mock_client.get_gex.side_effect = Exception("fail")
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    result = await fetcher.fetch_gex("AAPL")
    assert result == []


@pytest.mark.asyncio
async def test_fetch_trades(mock_client, mock_uw, mock_logger):
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    now = datetime.now(timezone.utc)
    trades, metrics = await fetcher.fetch_trades("AAPL", now, now)
    assert len(trades) == 1
    assert metrics["trades_fetch_fail"] == 0


@pytest.mark.asyncio
async def test_fetch_trades_error(mock_client, mock_uw, mock_logger):
    mock_client.get_trades.side_effect = Exception("fail")
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    now = datetime.now(timezone.utc)
    trades, metrics = await fetcher.fetch_trades("AAPL", now, now)
    assert trades == []
    assert metrics["trades_fetch_fail"] == 1


@pytest.mark.asyncio
async def test_fetch_quotes(mock_client, mock_uw, mock_logger):
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    now = datetime.now(timezone.utc)
    quotes, metrics = await fetcher.fetch_quotes("AAPL", now, now)
    assert len(quotes) == 1
    assert metrics["quotes_fetch_fail"] == 0


@pytest.mark.asyncio
async def test_fetch_quotes_error(mock_client, mock_uw, mock_logger):
    mock_client.get_quotes.side_effect = Exception("fail")
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    now = datetime.now(timezone.utc)
    quotes, metrics = await fetcher.fetch_quotes("AAPL", now, now)
    assert quotes == []
    assert metrics["quotes_fetch_fail"] == 1


@pytest.mark.asyncio
async def test_fetch_bars_uses_cache(mock_client, mock_uw, mock_logger):
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    now = datetime.now(timezone.utc)
    start = now - __import__("datetime").timedelta(hours=1)
    bars1, m1 = await fetcher.fetch_bars("AAPL", start, now)
    assert len(bars1) == 1
    # Second call should use cache (incremental fetch)
    bars2, m2 = await fetcher.fetch_bars("AAPL", start, now)
    assert m2["cache_hits"] == 1


def test_get_historical_bars_sync(mock_client, mock_uw, mock_logger):
    fetcher = DataFetcher(mock_client, mock_uw, mock_logger)
    now = datetime.now(timezone.utc)
    result = fetcher._get_historical_bars_sync("aapl", now, now, "1Min")
    mock_client.get_historical_bars.assert_called_once_with("AAPL", now, now, "1Min")
    assert "bars" in result
