from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from scripts.smoke_gateway_heber_integration import (
    SmokeConfig,
    check_gateway_authenticated,
    check_gateway_sink_ready,
    check_heber_silver_partition,
)


def _make_client(handler) -> httpx.Client:
    return httpx.Client(
        base_url="http://gateway.test",
        transport=httpx.MockTransport(handler),
        timeout=5.0,
    )


def test_check_gateway_authenticated_uses_gateway_key_header() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["key"] = request.headers.get("X-Gateway-Key")
        return httpx.Response(200, json={"success": True, "data": {"most_actives": []}})

    ok, detail = check_gateway_authenticated(
        client=_make_client(handler),
        gateway_key="gw_test_key",
    )

    assert ok is True
    assert "ok" in detail.lower()
    assert seen["path"] == "/api/v1/alpaca/screener/most-actives"
    assert seen["key"] == "gw_test_key"


def test_check_gateway_sink_ready_requires_sink_when_enabled() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ready", "checks": {"cache": "ok"}})

    ok, detail = check_gateway_sink_ready(
        client=_make_client(handler),
        require_sink=True,
    )

    assert ok is False
    assert "sinks" in detail


def test_check_heber_silver_partition_detects_parquet_file(tmp_path: Path) -> None:
    silver_file = tmp_path / "silver" / "feed=bars" / "instrument_type=equity" / "dt=2026-02-10" / "part-1.parquet"
    silver_file.parent.mkdir(parents=True, exist_ok=True)
    silver_file.write_bytes(b"PAR1")

    config = SmokeConfig(
        gateway_url="http://gateway.test",
        gateway_key="gw_test_key",
        heber_catalog_url="http://heber.test/api/v1",
        heber_data_root=tmp_path,
        smoke_symbol="AAPL",
        timeout_seconds=5.0,
        require_sink=True,
        require_silver_file=True,
        required_dataset="bars",
    )

    ok, detail = check_heber_silver_partition(config)
    assert ok is True
    assert "parquet" in detail.lower()
