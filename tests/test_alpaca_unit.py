import asyncio
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock alpaca modules before importing AlpacaClient
sys.modules["alpaca"] = MagicMock()
sys.modules["alpaca.trading"] = MagicMock()
sys.modules["alpaca.trading.client"] = MagicMock()
sys.modules["alpaca.trading.requests"] = MagicMock()
sys.modules["alpaca.trading.enums"] = MagicMock()
sys.modules["alpaca.data"] = MagicMock()
sys.modules["alpaca.data.historical"] = MagicMock()
sys.modules["alpaca.trading.stream"] = MagicMock()

from src.data.alpaca import AlpacaClient  # noqa


@pytest.fixture
def mock_config():
    loader = MagicMock()
    loader.get_env.side_effect = lambda k, d=None: "mock_key" if "KEY" in k else d
    return loader


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def client(mock_config, mock_logger):
    with (
        patch("src.data.alpaca.TradingClient") as mock_trading,
        patch("src.data.alpaca.StockHistoricalDataClient") as mock_historical,
    ):
        client = AlpacaClient(mock_config, mock_logger)
        # Ensure mocks are attached
        client.trading_client = mock_trading.return_value
        client.historical_client = mock_historical.return_value
        yield client


@pytest.mark.unit
def test_to_bar_conversion(client):
    # Test dictionary input
    data_dict = {
        "t": datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc),
        "o": 100,
        "h": 105,
        "l": 95,
        "c": 102,
        "v": 1000,
        "symbol": "AAPL",
    }
    bar = client._to_bar(data_dict)
    assert bar.symbol == "AAPL"
    assert bar.open == pytest.approx(100.0)
    assert bar.volume == pytest.approx(1000.0)

    # Test object input
    data_obj = MagicMock()
    data_obj.timestamp = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
    data_obj.open = 200
    data_obj.high = 205
    data_obj.low = 195
    data_obj.close = 202
    data_obj.volume = 500
    bar2 = client._to_bar(data_obj, symbol_override="TSLA")
    assert bar2.symbol == "TSLA"
    assert bar2.close == pytest.approx(202.0)


@pytest.mark.unit
def test_get_historical_bars_success(client):
    symbol = "SPY"
    start = datetime.now(timezone.utc)
    end = datetime.now(timezone.utc)

    # Mock historical client response
    mock_resp = MagicMock()
    # Data structure: {symbol: [bar1, bar2]}
    mock_bar = MagicMock()
    mock_bar.timestamp = start
    mock_bar.open = 150.0
    mock_bar.high = 151.0
    mock_bar.low = 149.0
    mock_bar.close = 150.5
    mock_bar.volume = 100000.0

    mock_resp.data = {symbol: [mock_bar]}

    # Verify mock structure before setting return_value
    assert isinstance(client.historical_client, MagicMock)
    # Configure return value
    client.historical_client.get_stock_bars.return_value = mock_resp

    res = client.get_historical_bars(symbol, start, end)

    assert "bars" in res
    assert len(res["bars"]) == 1
    assert res["bars"][0]["c"] == pytest.approx(150.5)

    client.historical_client.get_stock_bars.assert_called_once()


@pytest.mark.unit
def test_subscribe_queues_if_stream_not_started(client):
    client.get_stream_client = MagicMock()
    # Not calling start_stream yet

    client.subscribe("AAPL")
    assert "AAPL" in client._pending_symbols
    assert "AAPL" not in client._subscribed_symbols

    # Verify warning
    client.logger.warning.assert_called()


@pytest.mark.unit
def test_subscribe_immediate_if_handler_exists(client):
    # Simulate started stream
    client.get_stream_client = MagicMock()
    client._bar_handler = MagicMock()
    stream_mock = client.get_stream_client()

    client.subscribe("MSFT")

    stream_mock.subscribe_bars.assert_called_with(client._bar_handler, "MSFT")
    assert "MSFT" in client._subscribed_symbols
    assert "MSFT" not in client._pending_symbols


@pytest.mark.unit
def test_unsubscribe_success(client):
    stream_mock = MagicMock()
    client.get_stream_client = MagicMock(return_value=stream_mock)
    client._subscribed_symbols.add("AAPL")

    client.unsubscribe("AAPL")

    stream_mock.unsubscribe_bars.assert_called_with("AAPL")
    assert "AAPL" not in client._subscribed_symbols


@pytest.mark.unit
def test_unsubscribe_handles_keyerror(client):
    stream_mock = MagicMock()
    client.get_stream_client = MagicMock(return_value=stream_mock)
    stream_mock.unsubscribe_bars.side_effect = KeyError("Symbol not found")

    client.unsubscribe("AAPL")

    client.logger.warning.assert_called()
    assert "AAPL" not in client._subscribed_symbols


@pytest.mark.unit
def test_inspect_arity(client):
    def cb0():
        pass  # No-op

    def cb1(bar):
        pass  # No-op

    def cb2(symbol, bar):
        pass  # No-op

    def cb3(ctx, symbol, bar):
        pass  # No-op

    assert client._inspect_arity(cb0) == 0
    assert client._inspect_arity(cb1) == 1
    assert client._inspect_arity(cb2) == 2
    assert client._inspect_arity(cb3) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invoke_bar_callback(client):
    # Test arity 1
    cb1 = AsyncMock()
    bar = MagicMock()
    await client._invoke_bar_callback(cb1, bar, "AAPL", 1)
    cb1.assert_awaited_with(bar)

    # Test arity 2
    cb2 = AsyncMock()
    await client._invoke_bar_callback(cb2, bar, "AAPL", 2)
    cb2.assert_awaited_with("AAPL", bar)

    # Test fallback
    cb_fallback = AsyncMock()
    # If arity matches neither, it tries arity 1 first.
    await client._invoke_bar_callback(cb_fallback, bar, "AAPL", -1)
    cb_fallback.assert_awaited_with(bar)


@pytest.mark.unit
def test_make_bar_handler(client):
    cb = MagicMock()
    # Arity 1
    handler = client._make_bar_handler(cb)

    # Invoke handler
    # handler is async code?
    # lines 219: async def on_bar_wrapper(data):

    data = MagicMock()
    data.symbol = "AAPL"
    data.timestamp = datetime.now(timezone.utc)
    data.open = 100
    data.high = 105
    data.low = 95
    data.close = 102
    data.volume = 1000

    import asyncio

    asyncio.run(handler(data))

    cb.assert_called()  # because inspect_arity(MagicMock) might fail or guess.
    # MagicMock signature usually empty?

    # To be precise, let's use a real function for callback
    real_cb = MagicMock()

    def wrapper(bar):
        real_cb(bar)

    handler2 = client._make_bar_handler(wrapper)
    asyncio.run(handler2(data))
    real_cb.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_stream_runs_backoff_loop(client):
    # Mock get_stream_client
    # stream_mock = MagicMock() # Unused
    # run must be blocking, so we need to mock it effectively.
    # _run_stream_with_backoff calls asyncio.to_thread(stream.run)
    # We want to break the loop or run once.

    # We can mock _run_stream_with_backoff to verified it is called
    with patch.object(client, "_run_stream_with_backoff", new_callable=AsyncMock) as mock_run:
        cb = MagicMock()
        await client.start_stream(cb)

        mock_run.assert_awaited_once()
        assert client._bar_handler is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_stream_with_backoff_retries(client):
    stream = MagicMock()
    # We need to simulate stream.run raising exception then succeeding or cancelled
    # It runs in a loop.
    # Logic:
    # try: await asyncio.to_thread(stream.run)
    # except Exception: retry

    # We can't easily break an infinite loop in the test unless we raise CancelledError eventually

    # Side effect: first call raises RuntimeError, second call raises CancelledError
    stream.run.side_effect = [RuntimeError("Fail"), asyncio.CancelledError("Stop")]

    # Reduce sleep time
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        try:
            await client._run_stream_with_backoff(stream, on_reconnect=None)
        except asyncio.CancelledError:
            pass

        assert stream.run.call_count == 2
        mock_sleep.assert_awaited()
