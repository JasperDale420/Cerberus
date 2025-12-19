from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import httpx
import pytest

from src.data.unusual_whales import UnusualWhalesClient


class _Cfg:
    def __init__(self, url_template: str = "") -> None:
        self._url_template = url_template

    def get_env(self, _key: str, default: Optional[str] = None) -> str:
        return default or ""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unusual_whales_client_parses_list_and_dict_shapes() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("date") == "2025-01-01"
        if request.url.path.endswith("/AAPL"):
            return httpx.Response(200, json=[{"x": 1}])
        if request.url.path.endswith("/MSFT"):
            return httpx.Response(200, json={"data": [{"y": 2}]})
        return httpx.Response(200, json={"trades": [{"z": 3}]})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://uw.test")

    cfg = _Cfg()
    logger = MagicMock()
    uw = UnusualWhalesClient(
        cfg,  # type: ignore
        logger,
        client=client,
        config={"unusual_whales": {"flow_url_template": "https://uw.test/{symbol}"}},
    )  # type: ignore

    assert await uw.get_option_flow("aapl", "2025-01-01") == [{"x": 1}]
    assert await uw.get_option_flow("msft", "2025-01-01") == [{"y": 2}]
    assert await uw.get_option_flow("tsla", "2025-01-01") == [{"z": 3}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unusual_whales_client_degrades_to_empty_on_http_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://uw.test")

    cfg = _Cfg()
    logger = MagicMock()
    uw = UnusualWhalesClient(
        cfg,  # type: ignore
        logger,
        client=client,
        config={"unusual_whales": {"flow_url_template": "https://uw.test/{symbol}"}},
    )  # type: ignore

    out = await uw.get_option_flow("AAPL", "2025-01-01")
    assert out == []
    logger.warning.assert_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unusual_whales_client_warns_on_unexpected_payload_shape() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://uw.test")

    cfg = MagicMock()
    cfg.get_env.return_value = "https://uw.test/{symbol}"  # env override
    logger = MagicMock()
    uw = UnusualWhalesClient(
        cfg,
        logger,
        client=client,
        config={"unusual_whales": {"flow_url_template": "ignored"}},
    )

    out = await uw.get_option_flow("AAPL", "2025-01-01")
    assert out == []
    logger.warning.assert_called()
