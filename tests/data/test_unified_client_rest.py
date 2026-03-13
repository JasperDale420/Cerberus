"""Tests for UnifiedDataClient REST methods."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.data.client import UnifiedDataClient

# ---------------------------------------------------------------------------
# Initialization & URL construction
# ---------------------------------------------------------------------------


def test_client_initializes():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="key")
    assert c.gateway_url == "http://test:8080"
    assert c.gateway_key == "key"
    c.close()


def test_ws_url_construction_http():
    c = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="k")
    assert c._ws_url == "ws://localhost:8080/ws"
    c.close()


def test_ws_url_construction_https():
    c = UnifiedDataClient(gateway_url="https://gw.example.com", gateway_key="k")
    assert c._ws_url == "wss://gw.example.com/ws"
    c.close()


def test_ws_url_construction_trailing_slash():
    c = UnifiedDataClient(gateway_url="http://localhost:8080/", gateway_key="k")
    assert c._ws_url == "ws://localhost:8080/ws"
    c.close()


def test_default_timeout():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    assert c._timeout == 30.0
    c.close()


def test_custom_timeout():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k", timeout=10.0)
    assert c._timeout == 10.0
    c.close()


# ---------------------------------------------------------------------------
# Bar normalization
# ---------------------------------------------------------------------------


def test_normalize_bars_response_already_normalized():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    payload = {"bars": [{"t": "2026-01-01", "o": 1, "h": 2, "l": 0, "c": 1.5, "v": 100}]}
    assert c._normalize_bars_response(payload) == payload
    c.close()


def test_normalize_bars_response_gateway_envelope():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    payload = {
        "data": {"bars": [{"timestamp": "2026-01-01", "open": 1, "high": 2, "low": 0, "close": 1.5, "volume": 100}]}
    }
    result = c._normalize_bars_response(payload)
    assert "bars" in result
    assert result["bars"][0]["t"] == "2026-01-01"
    assert result["bars"][0]["o"] == 1
    assert result["bars"][0]["h"] == 2
    assert result["bars"][0]["l"] == 0
    assert result["bars"][0]["c"] == 1.5
    assert result["bars"][0]["v"] == 100
    c.close()


def test_normalize_bars_response_passthrough():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    payload = {"something_else": True}
    assert c._normalize_bars_response(payload) == payload
    c.close()


def test_normalize_bar_non_dict():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    assert c._normalize_bar("not_a_dict") == {}
    c.close()


def test_normalize_bar_short_keys_preferred():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    item = {"t": "ts1", "o": 10, "h": 20, "l": 5, "c": 15, "v": 999, "open": 0, "high": 0}
    result = c._normalize_bar(item)
    assert result["o"] == 10
    assert result["h"] == 20
    c.close()


# ---------------------------------------------------------------------------
# Trade normalization
# ---------------------------------------------------------------------------


def test_normalize_trades_response_direct():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    payload = {"trades": [{"t": "ts", "p": 100.0, "s": 50, "c": ["@"], "x": "V", "z": "A"}]}
    result = c._normalize_trades_response(payload)
    assert len(result) == 1
    assert result[0]["p"] == 100.0
    c.close()


def test_normalize_trades_response_gateway_envelope():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    payload = {
        "data": {
            "trades": [{"timestamp": "ts", "price": 55.0, "size": 10, "conditions": [], "exchange": "N", "tape": "C"}]
        }
    }
    result = c._normalize_trades_response(payload)
    assert len(result) == 1
    assert result[0]["t"] == "ts"
    assert result[0]["p"] == 55.0
    assert result[0]["s"] == 10
    assert result[0]["x"] == "N"
    assert result[0]["z"] == "C"
    c.close()


def test_normalize_trades_response_no_trades():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")
    assert c._normalize_trades_response({"data": {}}) == []
    c.close()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


def _make_client():
    return UnifiedDataClient(gateway_url="http://test:8080", gateway_key="k")


def test_retry_delay_exponential():
    c = _make_client()
    assert c._get_retry_delay_seconds(1) == 1.0
    assert c._get_retry_delay_seconds(2) == 2.0
    assert c._get_retry_delay_seconds(3) == 4.0
    c.close()


def test_retry_delay_respects_retry_after_header():
    c = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "5"}
    assert c._get_retry_delay_seconds(1, response=mock_resp) == 5.0
    c.close()


def test_is_retryable_status():
    c = _make_client()
    assert c._is_retryable_status(429) is True
    assert c._is_retryable_status(500) is True
    assert c._is_retryable_status(502) is True
    assert c._is_retryable_status(503) is True
    assert c._is_retryable_status(400) is False
    assert c._is_retryable_status(401) is False
    assert c._is_retryable_status(404) is False
    c.close()


@patch.object(UnifiedDataClient, "_get_retry_delay_seconds", return_value=0)
def test_request_with_retry_raises_immediately_on_401(mock_delay):
    c = _make_client()
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unauthorized", request=MagicMock(), response=mock_response
    )
    c.client = MagicMock()
    c.client.request.return_value = mock_response
    with pytest.raises(httpx.HTTPStatusError):
        c._request_with_retry("GET", "/test")
    assert c.client.request.call_count == 1
    c.close()


@patch("src.data.client.time.sleep")
def test_request_with_retry_retries_on_500(mock_sleep):
    c = _make_client()
    fail_resp = MagicMock(spec=httpx.Response)
    fail_resp.status_code = 500
    ok_resp = MagicMock(spec=httpx.Response)
    ok_resp.status_code = 200
    c.client = MagicMock()
    c.client.request.side_effect = [fail_resp, ok_resp]
    result = c._request_with_retry("GET", "/test")
    assert result == ok_resp
    assert c.client.request.call_count == 2
    c.close()


@patch("src.data.client.time.sleep")
def test_request_with_retry_retries_on_transport_error(mock_sleep):
    c = _make_client()
    ok_resp = MagicMock(spec=httpx.Response)
    ok_resp.status_code = 200
    c.client = MagicMock()
    c.client.request.side_effect = [httpx.TimeoutException("timeout"), ok_resp]
    result = c._request_with_retry("GET", "/test")
    assert result == ok_resp
    assert c.client.request.call_count == 2
    c.close()


# ---------------------------------------------------------------------------
# REST methods (mocked HTTP)
# ---------------------------------------------------------------------------


def _client_with_mock_request(response_json, status_code=200):
    c = _make_client()
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = response_json
    c.client = MagicMock()
    c.client.request.return_value = mock_resp
    return c


def test_get_historical_bars():
    payload = {"bars": [{"t": "2026-01-01", "o": 1, "h": 2, "l": 0, "c": 1.5, "v": 100}]}
    c = _client_with_mock_request(payload)
    result = c.get_historical_bars(
        "AAPL", datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)
    )
    assert result == payload
    call_args = c.client.request.call_args
    assert call_args[0] == ("GET", "/api/v1/alpaca/stocks/AAPL/bars")
    c.close()


def test_get_trades():
    payload = {"trades": [{"t": "ts", "p": 100, "s": 50, "c": [], "x": "V", "z": "A"}]}
    c = _client_with_mock_request(payload)
    result = c.get_trades("AAPL", datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert len(result) == 1
    assert result[0]["p"] == 100
    c.close()


def test_get_quotes():
    payload = {"quotes": [{"t": "ts", "bp": 100, "ap": 101}]}
    c = _client_with_mock_request(payload)
    result = c.get_quotes("AAPL", datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert result == payload
    c.close()


def test_get_snapshot():
    payload = {"data": {"latestTrade": {"p": 150}}}
    c = _client_with_mock_request(payload)
    result = c.get_snapshot("AAPL")
    assert result == payload
    c.close()


def test_get_flow():
    payload = {"data": [{"premium": 100000}]}
    c = _client_with_mock_request(payload)
    result = c.get_flow("AAPL")
    assert result == payload
    c.close()


def test_get_flow_with_date():
    payload = {"data": [{"premium": 100000}]}
    c = _client_with_mock_request(payload)
    c.get_flow("AAPL", date_str="2026-01-01")
    call_args = c.client.request.call_args
    assert call_args[1]["params"]["date"] == "2026-01-01"
    c.close()


def test_get_gex():
    payload = {"data": [{"strike": 150, "gex": 1000}]}
    c = _client_with_mock_request(payload)
    result = c.get_gex("AAPL")
    assert len(result) == 1
    assert result[0]["strike"] == 150
    c.close()


def test_get_gex_nested_rows():
    payload = {"data": {"rows": [{"strike": 150, "gex": 1000}]}}
    c = _client_with_mock_request(payload)
    result = c.get_gex("AAPL")
    assert len(result) == 1
    c.close()


def test_get_most_actives():
    payload = {"data": {"most_actives": [{"symbol": "AAPL"}, {"symbol": "TSLA"}]}}
    c = _client_with_mock_request(payload)
    result = c.get_most_actives(top=2)
    assert result == ["AAPL", "TSLA"]
    c.close()


def test_get_movers():
    payload = {"data": {"gainers": [{"symbol": "NVDA"}], "losers": [{"symbol": "META"}]}}
    c = _client_with_mock_request(payload)
    result = c.get_movers(top=1)
    assert result == {"gainers": ["NVDA"], "losers": ["META"]}
    c.close()


def test_submit_order():
    payload = {"data": {"id": "order-123", "status": "accepted"}}
    c = _client_with_mock_request(payload)
    result = c.submit_order("AAPL", "buy", qty=10)
    assert result["id"] == "order-123"
    call_args = c.client.request.call_args
    assert call_args[0][0] == "POST"
    c.close()


def test_get_orders():
    payload = {"data": [{"id": "o1"}, {"id": "o2"}]}
    c = _client_with_mock_request(payload)
    result = c.get_orders(status="open", limit=50)
    assert len(result) == 2
    c.close()


def test_cancel_order():
    payload = {"data": {"cancelled": True}}
    c = _client_with_mock_request(payload)
    result = c.cancel_order("order-123")
    assert result is True
    call_args = c.client.request.call_args
    assert call_args[0][0] == "DELETE"
    assert "order-123" in call_args[0][1]
    c.close()


def test_get_prior_day_stats():
    bars = [
        {"t": "2026-01-10T00:00:00Z", "o": 100, "h": 110, "l": 95, "c": 105, "v": 1000},
        {"t": "2026-01-11T00:00:00Z", "o": 105, "h": 115, "l": 100, "c": 112, "v": 2000},
        {"t": "2026-01-12T00:00:00Z", "o": 112, "h": 120, "l": 108, "c": 118, "v": 1500},
    ]
    payload = {"bars": bars}
    c = _client_with_mock_request(payload)
    current_time = datetime(2026, 1, 12, 14, 0, tzinfo=timezone.utc)
    high, low, close = c.get_prior_day_stats("AAPL", current_time)
    # Should return the second-to-last bar (last complete day before current)
    assert high == 115
    assert low == 100
    assert close == 112
    c.close()


def test_get_avg_daily_volume():
    bars = [{"t": f"2026-01-{i:02d}", "o": 1, "h": 2, "l": 0, "c": 1, "v": i * 100} for i in range(1, 6)]
    payload = {"bars": bars}
    c = _client_with_mock_request(payload)
    result = c.get_avg_daily_volume("AAPL", datetime(2026, 1, 6, tzinfo=timezone.utc), lookback_days=5)
    # avg of 100, 200, 300, 400, 500 = 300
    assert result == 300.0
    c.close()


def test_get_avg_daily_volume_empty():
    c = _client_with_mock_request({"bars": []})
    result = c.get_avg_daily_volume("AAPL", datetime(2026, 1, 6, tzinfo=timezone.utc))
    assert result == 0.0
    c.close()


def test_close():
    c = _make_client()
    mock_client = MagicMock()
    c.client = mock_client
    c.close()
    mock_client.close.assert_called_once()
