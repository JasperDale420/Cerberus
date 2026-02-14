from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.data.gateway_stream import GatewayStreamClient


def _cfg() -> MagicMock:
    cfg = MagicMock()

    def _env(key: str, default=None):
        if key in {"CERBERUS_GATEWAY_URL", "DATA_INGESTION_URL"}:
            return "http://localhost:8080"
        if key == "CERBERUS_GATEWAY_KEY":
            return "gw_test_key"
        return default

    cfg.get_env.side_effect = _env
    return cfg


def test_build_ws_url_from_http_base() -> None:
    client = GatewayStreamClient(_cfg(), MagicMock())
    assert client.ws_url == "ws://localhost:8080/ws"


def test_build_ws_url_from_https_base() -> None:
    assert GatewayStreamClient._build_ws_url("https://gateway.example.com") == "wss://gateway.example.com/ws"


def test_extract_bar_payload_prefers_data_field() -> None:
    client = GatewayStreamClient(_cfg(), MagicMock())
    msg = {
        "type": "data",
        "feed": "bars",
        "data": {"S": "AAPL", "t": "2026-02-13T14:30:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100},
    }
    payload = client._extract_bar_payload(msg)
    assert payload is not None
    assert payload["S"] == "AAPL"


def test_extract_bar_payload_falls_back_to_envelope_payload() -> None:
    client = GatewayStreamClient(_cfg(), MagicMock())
    msg = {
        "type": "data",
        "feed": "bars",
        "envelope": {
            "payload": {"S": "MSFT", "t": "2026-02-13T14:31:00Z", "o": 3, "h": 4, "l": 2.5, "c": 3.5, "v": 200}
        },
    }
    payload = client._extract_bar_payload(msg)
    assert payload is not None
    assert payload["S"] == "MSFT"


def test_normalize_bar_from_gateway_payload() -> None:
    client = GatewayStreamClient(_cfg(), MagicMock())
    bar = client._normalize_bar_from_data(
        {
            "S": "SPY",
            "t": "2026-02-13T14:32:00Z",
            "o": 500.0,
            "h": 501.0,
            "l": 499.5,
            "c": 500.5,
            "v": 12345,
        }
    )
    assert bar.symbol == "SPY"
    assert bar.open == 500.0
    assert bar.close == 500.5
    assert bar.time == datetime(2026, 2, 13, 14, 32, tzinfo=timezone.utc)


def test_normalize_bar_prefers_zero_and_fallback_values() -> None:
    client = GatewayStreamClient(_cfg(), MagicMock())
    bar = client._normalize_bar_from_data(
        {
            "S": "MSFT",
            "t": "2026-02-13T14:35:00Z",
            "o": 0.0,
            "h": None,
            "low": 99.5,
            "c": None,
            "v": None,
            "open": 500.0,
            "high": 501.0,
            "close": 500.5,
            "volume": 321.0,
        }
    )
    assert bar.open == 0.0
    assert bar.high == 501.0
    assert bar.low == 99.5
    assert bar.close == 500.5
    assert bar.volume == 321.0


@pytest.mark.asyncio
async def test_handle_message_invokes_callback_with_symbol_and_bar() -> None:
    client = GatewayStreamClient(_cfg(), MagicMock())
    callback = MagicMock()
    msg = {
        "type": "data",
        "feed": "bars",
        "data": {"S": "QQQ", "t": "2026-02-13T14:33:00Z", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 50},
    }

    await client._handle_message(callback, 2, msg)

    callback.assert_called_once()
    symbol, bar = callback.call_args.args
    assert symbol == "QQQ"
    assert bar.symbol == "QQQ"
