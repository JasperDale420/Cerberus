from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from scripts.smoke_gateway_heber_integration import (
    SmokeConfig,
    check_gateway_authenticated,
    check_gateway_sink_publish_activity,
    check_gateway_sink_ready,
    check_heber_layer_has_fresh_file,
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
        sink_metric_poll_attempts=2,
        sink_metric_poll_interval_seconds=0.0,
        write_poll_timeout_seconds=0.0,
        write_poll_interval_seconds=0.0,
    )

    ok, detail = check_heber_silver_partition(config)
    assert ok is True
    assert "parquet" in detail.lower()


def test_check_gateway_sink_publish_activity_observes_metrics_growth() -> None:
    seen: dict[str, Any] = {"metrics_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            seen["metrics_calls"] += 1
            seen["metrics_key"] = request.headers.get("X-Gateway-Key")
            value = 5 if seen["metrics_calls"] == 1 else 6
            return httpx.Response(
                200,
                text=(
                    "# HELP gateway_sink_publish_total Total data sink publish operations\n"
                    "# TYPE gateway_sink_publish_total counter\n"
                    f'gateway_sink_publish_total{{sink="redis_streams",topic="heber:events",status="success"}} {value}\n'
                ),
            )
        return httpx.Response(404)

    ok, detail = check_gateway_sink_publish_activity(
        client=_make_client(handler),
        gateway_key="gw_test_key",
        baseline_success_count=5.0,
        min_increment=1.0,
        poll_attempts=2,
        poll_interval_seconds=0.0,
    )

    assert ok is True
    assert "delta=" in detail
    assert seen["metrics_key"] == "gw_test_key"


def test_check_heber_layer_has_fresh_file_checks_modified_time(tmp_path: Path) -> None:
    silver_file = tmp_path / "silver" / "feed=bars" / "instrument_type=equity" / "dt=2026-02-10" / "part-2.parquet"
    silver_file.parent.mkdir(parents=True, exist_ok=True)
    silver_file.write_bytes(b"PAR1")

    baseline = datetime.now(UTC) - timedelta(seconds=10)
    fresh_mtime = baseline + timedelta(seconds=5)
    os.utime(silver_file, (fresh_mtime.timestamp(), fresh_mtime.timestamp()))

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
        sink_metric_poll_attempts=2,
        sink_metric_poll_interval_seconds=0.0,
        write_poll_timeout_seconds=0.0,
        write_poll_interval_seconds=0.0,
    )

    ok, detail = check_heber_layer_has_fresh_file(
        config=config,
        layer="silver",
        baseline_time=baseline,
        poll_timeout_seconds=0.0,
        poll_interval_seconds=0.0,
    )

    assert ok is True
    assert "fresh" in detail.lower()
