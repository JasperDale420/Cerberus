from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import src.data.alpaca as alpaca_mod
from src.core.errors import ErrorCode
from src.data.alpaca import AlpacaClient


@pytest.mark.unit
def test_alpaca_client_init_raises_when_missing_credentials(monkeypatch) -> None:
    logger = MagicMock()
    cfg = MagicMock()

    def _get_env(key: str, default=None):
        raise ValueError(f"Missing required environment variable: {key}")

    cfg.get_env.side_effect = _get_env

    with pytest.raises(ValueError):
        AlpacaClient(cfg, logger)

    logger.critical.assert_called()


@pytest.mark.unit
def test_get_account_delegates_and_raises_on_error(monkeypatch) -> None:
    logger = MagicMock()
    cfg = MagicMock()
    cfg.get_env.side_effect = lambda k, default=None: {
        "ALPACA_API_KEY": "k",
        "ALPACA_SECRET_KEY": "s",
        "ALPACA_PAPER": "True",
    }.get(k, default)

    monkeypatch.setattr(alpaca_mod, "TradingClient", MagicMock())
    monkeypatch.setattr(alpaca_mod, "StockHistoricalDataClient", MagicMock())

    c = AlpacaClient(cfg, logger)
    c.trading_client.get_account.return_value = {"ok": True}  # type: ignore[attr-defined]
    assert c.get_account() == {"ok": True}

    c.trading_client.get_account.side_effect = RuntimeError("boom")  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        c.get_account()
    logger.error.assert_called()
    _, kwargs = logger.error.call_args
    assert kwargs.get("error_code") == ErrorCode.ALPACA_ACCOUNT_FETCH_FAILED.value


@pytest.mark.unit
def test_get_historical_bars_uses_alpaca_historical_client(monkeypatch) -> None:
    logger = MagicMock()
    cfg = MagicMock()
    cfg.get_env.side_effect = lambda k, default=None: {
        "ALPACA_API_KEY": "k",
        "ALPACA_SECRET_KEY": "s",
        "ALPACA_PAPER": "True",
    }.get(k, default)

    monkeypatch.setattr(alpaca_mod, "TradingClient", MagicMock())
    hist = MagicMock()
    monkeypatch.setattr(
        alpaca_mod, "StockHistoricalDataClient", MagicMock(return_value=hist)
    )

    c = AlpacaClient(cfg, logger)
    bar = MagicMock()
    bar.open = 1.0
    bar.high = 2.0
    bar.low = 0.5
    bar.close = 1.5
    bar.volume = 123
    bar.timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)

    resp = MagicMock()
    resp.data = {"AAPL": [bar]}
    hist.get_stock_bars.return_value = resp

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 2, tzinfo=timezone.utc)
    out = c.get_historical_bars("AAPL", start, end, timeframe="1Min")

    assert out["bars"][0]["c"] == 1.5
    assert hist.get_stock_bars.call_count == 1
    req = hist.get_stock_bars.call_args[0][0]
    assert req.symbol_or_symbols == "AAPL"
    assert req.start == start.replace(tzinfo=None)
    assert req.end == end.replace(tzinfo=None)


@pytest.mark.unit
def test_stream_client_cached_and_subscribe_unsubscribe(monkeypatch) -> None:
    logger = MagicMock()
    cfg = MagicMock()
    cfg.get_env.side_effect = lambda k, default=None: {
        "ALPACA_API_KEY": "k",
        "ALPACA_SECRET_KEY": "s",
        "ALPACA_PAPER": "True",
    }.get(k, default)

    monkeypatch.setattr(alpaca_mod, "TradingClient", MagicMock())
    monkeypatch.setattr(alpaca_mod, "StockHistoricalDataClient", MagicMock())

    stream = MagicMock()
    monkeypatch.setattr(alpaca_mod, "StockDataStream", MagicMock(return_value=stream))

    c = AlpacaClient(cfg, logger)

    s1 = c.get_stream_client()
    s2 = c.get_stream_client()
    assert s1 is s2

    # Subscribe before start_stream sets handler should warn.
    c.subscribe("AAPL")
    logger.warning.assert_called()

    # With handler, subscribe should call SDK.
    c._bar_handler = MagicMock()  # type: ignore[attr-defined]
    c.subscribe("AAPL")
    stream.subscribe_bars.assert_called_with(c._bar_handler, "AAPL")  # type: ignore[arg-type]

    c.unsubscribe("AAPL")
    stream.unsubscribe_bars.assert_called_with("AAPL")
