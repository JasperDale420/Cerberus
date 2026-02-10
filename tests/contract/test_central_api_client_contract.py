from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from src.data.api_client import CentralApiClient


def _make_client(handler) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(
        base_url="http://central.test", transport=transport, timeout=5.0
    )


@pytest.mark.contract
def test_get_alpaca_bars_contract_builds_correct_path_and_params() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["query"] = dict(request.url.params)
        seen["gateway_key"] = request.headers.get("X-Gateway-Key")
        return httpx.Response(200, json={"success": True, "data": {"bars": []}})

    cfg = MagicMock()
    cfg.get_env.side_effect = (
        lambda key, default=None: "gw_test_key"
        if key == "CERBERUS_GATEWAY_KEY"
        else "http://central.test"
        if key in {"CERBERUS_GATEWAY_URL", "DATA_INGESTION_URL", "CENTRAL_LLM_API_URL"}
        else default
    )
    logger = MagicMock()

    c = CentralApiClient(cfg, logger)
    c.client = _make_client(handler)
    c.client.headers["X-Gateway-Key"] = "gw_test_key"

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 2, tzinfo=timezone.utc)
    out = c.get_alpaca_bars("AAPL", start=start, end=end, timeframe="1Min")

    assert out == {"bars": []}
    assert seen["method"] == "GET"
    assert seen["url"].startswith("http://central.test/api/v1/alpaca/stocks/AAPL/bars")
    assert seen["query"]["timeframe"] == "1Min"
    assert "start" in seen["query"] and "end" in seen["query"]
    assert seen["gateway_key"] == "gw_test_key"


@pytest.mark.contract
def test_get_uw_flow_contract_calls_expected_path() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["gateway_key"] = request.headers.get("X-Gateway-Key")
        return httpx.Response(200, json={"data": []})

    cfg = MagicMock()
    cfg.get_env.side_effect = (
        lambda key, default=None: "gw_test_key"
        if key == "CERBERUS_GATEWAY_KEY"
        else "http://central.test"
        if key in {"CERBERUS_GATEWAY_URL", "DATA_INGESTION_URL", "CENTRAL_LLM_API_URL"}
        else default
    )
    logger = MagicMock()

    c = CentralApiClient(cfg, logger)
    c.client = _make_client(handler)
    c.client.headers["X-Gateway-Key"] = "gw_test_key"

    out = c.get_uw_flow("SPY")
    assert out == {"data": []}
    assert seen["url"].startswith("http://central.test/api/v1/uw/flow/SPY")
    assert seen["gateway_key"] == "gw_test_key"


@pytest.mark.contract
def test_chat_completion_contract_uses_openai_like_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content.decode("utf-8") or "{}")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    cfg = MagicMock()
    cfg.get_env.side_effect = (
        lambda key, default=None: "http://central.test"
        if key in {"CERBERUS_GATEWAY_URL", "DATA_INGESTION_URL", "CENTRAL_LLM_API_URL"}
        else default
    )
    logger = MagicMock()

    c = CentralApiClient(cfg, logger)
    c.client = _make_client(handler)

    out = c.chat_completion("model-x", [{"role": "user", "content": "hi"}])
    assert out["choices"][0]["message"]["content"] == "ok"
    assert seen["method"] == "POST"
    assert seen["url"].startswith("http://central.test/v1/chat/completions")
    assert seen["json"]["model"] == "model-x"
    assert seen["json"]["messages"][0]["role"] == "user"


@pytest.mark.contract
@pytest.mark.parametrize(
    "method_name,args",
    [
        ("get_alpaca_bars", ("AAPL",)),
        ("get_uw_flow", ("SPY",)),
        ("chat_completion", ("m", [{"role": "user", "content": "x"}])),
    ],
)
def test_central_api_client_raises_on_http_error(
    method_name: str, args: tuple[Any, ...]
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    cfg = MagicMock()
    cfg.get_env.side_effect = (
        lambda key, default=None: "http://central.test"
        if key in {"CERBERUS_GATEWAY_URL", "DATA_INGESTION_URL", "CENTRAL_LLM_API_URL"}
        else default
    )
    logger = MagicMock()

    c = CentralApiClient(cfg, logger)
    c.client = _make_client(handler)

    fn = getattr(c, method_name)
    with pytest.raises(httpx.HTTPError):
        fn(*args)
