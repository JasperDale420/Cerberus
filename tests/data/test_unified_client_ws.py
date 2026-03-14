"""Tests for UnifiedDataClient WebSocket streaming methods."""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.data.client import FEED_NAME_MAP, StreamQuote, StreamTrade, UnifiedDataClient


def _make_connect_mock(mock_ws):
    """Create a mock for websockets.connect that returns an awaitable resolving to mock_ws."""
    future = asyncio.Future()
    future.set_result(mock_ws)

    def connect_fn(*args, **kwargs):
        f = asyncio.Future()
        f.set_result(mock_ws)
        return f

    return connect_fn


@pytest.mark.asyncio
async def test_connect_authenticates():
    client = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="gw_test")
    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock(return_value=json.dumps({"type": "auth_result", "status": "ok"}))

    with patch("websockets.connect", side_effect=_make_connect_mock(mock_ws)):
        await client.connect()
    mock_ws.send.assert_called_once()
    sent = json.loads(mock_ws.send.call_args[0][0])
    assert sent["action"] == "auth"
    assert sent["key"] == "gw_test"
    assert client._ws is mock_ws


@pytest.mark.asyncio
async def test_connect_auth_failure_raises():
    client = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="bad_key")
    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock(return_value=json.dumps({"type": "auth_result", "status": "error"}))

    with patch("websockets.connect", side_effect=_make_connect_mock(mock_ws)):
        with pytest.raises(RuntimeError, match="auth failed"):
            await client.connect()
    assert client._ws is None


@pytest.mark.asyncio
async def test_subscribe_maps_feed_names():
    client = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="k")
    client._ws = AsyncMock()
    await client.subscribe(feeds=["bars", "quotes"], symbols=["AAPL", "msft"])
    client._ws.send.assert_called_once()
    sent = json.loads(client._ws.send.call_args[0][0])
    assert sent["action"] == "subscribe"
    assert set(sent["feeds"]) == {"stock_bars", "stock_quotes"}
    assert set(sent["symbols"]) == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_update_subscriptions_sends_deltas():
    client = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="k")
    client._ws = AsyncMock()
    client._subscribed_symbols = {"AAPL", "MSFT"}
    client._subscribed_feeds = {"stock_bars"}
    await client.update_subscriptions(["MSFT", "TSLA"])
    assert client._ws.send.call_count == 2


@pytest.mark.asyncio
async def test_disconnect_clears_state():
    client = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="k")
    client._ws = AsyncMock()
    client._subscribed_symbols = {"AAPL"}
    client._subscribed_feeds = {"stock_bars"}
    await client.disconnect()
    assert client._ws is None
    assert len(client._subscribed_symbols) == 0


def test_parse_timestamp_iso_string():
    ts = UnifiedDataClient._parse_timestamp("2026-03-13T10:30:00Z")
    assert ts.year == 2026
    assert ts.tzinfo is not None


def test_parse_timestamp_unix():
    ts = UnifiedDataClient._parse_timestamp(1773580200.0)
    assert isinstance(ts, datetime)


def test_parse_timestamp_datetime_naive():
    dt = datetime(2026, 3, 13, 10, 30, 0)
    ts = UnifiedDataClient._parse_timestamp(dt)
    assert ts.tzinfo == timezone.utc


def test_parse_timestamp_datetime_aware():
    dt = datetime(2026, 3, 13, 10, 30, 0, tzinfo=timezone.utc)
    ts = UnifiedDataClient._parse_timestamp(dt)
    assert ts.tzinfo == timezone.utc


def test_parse_timestamp_invalid_raises():
    with pytest.raises(ValueError, match="Invalid timestamp"):
        UnifiedDataClient._parse_timestamp(None)


def test_normalize_stream_bar():
    client = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    bar = client._normalize_stream_bar(
        {
            "S": "AAPL",
            "t": "2026-03-13T10:30:00Z",
            "o": 150.0,
            "h": 151.0,
            "l": 149.0,
            "c": 150.5,
            "v": 1000.0,
        }
    )
    assert bar.symbol == "AAPL"
    assert bar.close == 150.5
    assert bar.open == 150.0
    assert bar.volume == 1000.0


def test_normalize_stream_bar_long_keys():
    client = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    bar = client._normalize_stream_bar(
        {
            "symbol": "msft",
            "timestamp": "2026-03-13T10:30:00Z",
            "open": 400.0,
            "high": 405.0,
            "low": 399.0,
            "close": 402.0,
            "volume": 5000.0,
        }
    )
    assert bar.symbol == "MSFT"
    assert bar.high == 405.0


def test_normalize_stream_bar_missing_symbol_raises():
    client = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    with pytest.raises(ValueError, match="Missing symbol"):
        client._normalize_stream_bar({"t": "2026-03-13T10:30:00Z", "o": 1.0})


def test_normalize_stream_quote():
    client = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    quote = client._normalize_stream_quote(
        {
            "S": "AAPL",
            "t": "2026-03-13T10:30:00Z",
            "bp": 150.0,
            "bs": 100.0,
            "ap": 150.5,
            "as": 200.0,
        }
    )
    assert quote.symbol == "AAPL"
    assert quote.bid_price == 150.0
    assert quote.ask_size == 200.0
    assert isinstance(quote, StreamQuote)


def test_normalize_stream_trade():
    client = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    trade = client._normalize_stream_trade(
        {
            "S": "TSLA",
            "t": "2026-03-13T10:30:00Z",
            "p": 250.0,
            "s": 50.0,
            "c": ["@", "T"],
        }
    )
    assert trade.symbol == "TSLA"
    assert trade.price == 250.0
    assert trade.conditions == ["@", "T"]
    assert isinstance(trade, StreamTrade)


def test_extract_data_payload():
    feed, payload = UnifiedDataClient._extract_data_payload(
        {
            "type": "data",
            "feed": "bars",
            "data": {"S": "AAPL", "o": 150.0},
        }
    )
    assert feed == "bars"
    assert payload["S"] == "AAPL"


def test_extract_data_payload_envelope():
    feed, payload = UnifiedDataClient._extract_data_payload(
        {
            "type": "data",
            "feed": "bars",
            "envelope": {"payload": {"S": "AAPL", "o": 150.0}},
        }
    )
    assert feed == "bars"
    assert payload["S"] == "AAPL"


def test_extract_data_payload_not_data():
    feed, payload = UnifiedDataClient._extract_data_payload({"type": "heartbeat"})
    assert feed == ""
    assert payload is None


def test_feed_name_map_completeness():
    assert "bars" in FEED_NAME_MAP
    assert "quotes" in FEED_NAME_MAP
    assert "trades" in FEED_NAME_MAP


@pytest.mark.asyncio
async def test_unsubscribe():
    client = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="k")
    client._ws = AsyncMock()
    client._subscribed_symbols = {"AAPL", "MSFT"}
    await client.unsubscribe(feeds=["bars"], symbols=["AAPL"])
    client._ws.send.assert_called_once()
    sent = json.loads(client._ws.send.call_args[0][0])
    assert sent["action"] == "unsubscribe"
    assert "AAPL" not in client._subscribed_symbols
    assert "MSFT" in client._subscribed_symbols


@pytest.mark.asyncio
async def test_subscribe_no_op_without_ws():
    client = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="k")
    client._ws = None
    await client.subscribe(feeds=["bars"], symbols=["AAPL"])
    # Should not raise


@pytest.mark.asyncio
async def test_dispatch_event_bar():
    client = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    received = []

    def on_bar(symbol, bar):
        received.append((symbol, bar))

    await client._dispatch_event(
        "bars",
        {"S": "AAPL", "t": "2026-03-13T10:30:00Z", "o": 150.0, "h": 151.0, "l": 149.0, "c": 150.5, "v": 1000.0},
        on_bar=on_bar,
        on_quote=None,
        on_trade=None,
    )
    assert len(received) == 1
    assert received[0][0] == "AAPL"


@pytest.mark.asyncio
async def test_dispatch_event_quote_async():
    client = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    received = []

    async def on_quote(symbol, quote):
        received.append((symbol, quote))

    await client._dispatch_event(
        "quotes",
        {"S": "AAPL", "t": "2026-03-13T10:30:00Z", "bp": 150.0, "bs": 100.0, "ap": 150.5, "as": 200.0},
        on_bar=None,
        on_quote=on_quote,
        on_trade=None,
    )
    assert len(received) == 1
    assert isinstance(received[0][1], StreamQuote)


@pytest.mark.asyncio
async def test_dispatch_event_trade():
    client = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    received = []

    def on_trade(symbol, trade):
        received.append((symbol, trade))

    await client._dispatch_event(
        "trades",
        {"S": "TSLA", "t": "2026-03-13T10:30:00Z", "p": 250.0, "s": 50.0},
        on_bar=None,
        on_quote=None,
        on_trade=on_trade,
    )
    assert len(received) == 1
    assert received[0][0] == "TSLA"


@pytest.mark.asyncio
async def test_dispatch_event_swallows_callback_errors():
    client = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")

    def on_bar(symbol, bar):
        raise ValueError("boom")

    # Should not raise
    await client._dispatch_event(
        "bars",
        {"S": "AAPL", "t": "2026-03-13T10:30:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0},
        on_bar=on_bar,
        on_quote=None,
        on_trade=None,
    )
