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
    run_smoke,
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
        seen["cache"] = request.headers.get("X-Gateway-Cache")
        seen["cache_buster"] = request.url.params.get("cache_buster")
        return httpx.Response(200, json={"success": True, "data": {"most_actives": []}})

    ok, detail = check_gateway_authenticated(
        client=_make_client(handler),
        gateway_key="gw_test_key",
        smoke_symbol="AAPL",
    )

    assert ok is True
    assert "ok" in detail.lower()
    assert seen["path"] == "/api/v1/alpaca/stocks/AAPL/bars"
    assert seen["key"] == "gw_test_key"
    assert seen["cache"] == "bypass"
    assert seen["cache_buster"] is not None


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


def test_check_heber_bronze_layer_supports_provider_partition_layout(tmp_path: Path) -> None:
    bronze_file = tmp_path / "bronze" / "provider=alpaca" / "feed=bars" / "dt=2026-02-12" / "part-1.jsonl.gz"
    bronze_file.parent.mkdir(parents=True, exist_ok=True)
    bronze_file.write_bytes(b"{}")

    baseline = datetime.now(UTC) - timedelta(seconds=10)
    fresh_mtime = baseline + timedelta(seconds=5)
    os.utime(bronze_file, (fresh_mtime.timestamp(), fresh_mtime.timestamp()))

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
        layer="bronze",
        baseline_time=baseline,
        poll_timeout_seconds=0.0,
        poll_interval_seconds=0.0,
    )

    assert ok is True
    assert "fresh" in detail.lower()


def test_smoke_config_defaults_required_dataset_to_bars(monkeypatch) -> None:
    monkeypatch.delenv("CERBERUS_SMOKE_REQUIRED_DATASET", raising=False)
    config = SmokeConfig.from_env()
    assert config.required_dataset == "bars"


def test_run_smoke_emits_explicit_message_when_sink_ready_but_no_recent_writes(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
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
        sink_metric_poll_attempts=1,
        sink_metric_poll_interval_seconds=0.0,
        write_poll_timeout_seconds=0.0,
        write_poll_interval_seconds=0.0,
    )

    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration._read_sink_publish_success_total",
        lambda **_: (True, 10.0, "metrics ok"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_gateway_authenticated",
        lambda **_: (True, "gateway authenticated call ok"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_gateway_sink_ready",
        lambda **_: (True, "gateway readiness and sink checks ok"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_gateway_sink_publish_activity",
        lambda **_: (True, "sink publish activity observed"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_heber_catalog",
        lambda *_, **__: (True, "heber catalog datasets endpoint ok"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_heber_silver_partition",
        lambda *_: (True, "Silver parquet found"),
    )

    def _stale_layer(**kwargs):
        layer = kwargs["layer"]
        return False, f"{layer} latest file is stale"

    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_heber_layer_has_fresh_file",
        _stale_layer,
    )

    code = run_smoke(config)
    output = capsys.readouterr().out

    assert code == 1
    assert "Gateway sink is ready but no recent Heber writes were observed" in output


def test_run_smoke_uses_fresh_write_fallback_when_sink_counter_does_not_move(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
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
        sink_metric_poll_attempts=1,
        sink_metric_poll_interval_seconds=0.0,
        write_poll_timeout_seconds=0.0,
        write_poll_interval_seconds=0.0,
    )

    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration._read_sink_publish_success_total",
        lambda **_: (True, 10.0, "metrics ok"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_gateway_authenticated",
        lambda **_: (True, "gateway authenticated call ok"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_gateway_sink_ready",
        lambda **_: (True, "gateway readiness and sink checks ok"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_gateway_sink_publish_activity",
        lambda **_: (False, "counter unchanged"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_heber_catalog",
        lambda *_, **__: (True, "heber catalog datasets endpoint ok"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_heber_silver_partition",
        lambda *_: (True, "Silver parquet found"),
    )
    monkeypatch.setattr(
        "scripts.smoke_gateway_heber_integration.check_heber_layer_has_fresh_file",
        lambda **kwargs: (True, f"{kwargs['layer']} fresh file found"),
    )

    code = run_smoke(config)
    output = capsys.readouterr().out

    assert code == 0
    assert "gateway sink counter unchanged, but fresh Heber writes were observed" in output
