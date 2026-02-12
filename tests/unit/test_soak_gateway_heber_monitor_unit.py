from __future__ import annotations

from datetime import UTC, datetime

from scripts.soak_gateway_heber_monitor import (
    PollSnapshot,
    SoakConfig,
    _collect_snapshot,
    _default_gateway_key,
    _extract_sink_publish_success_total,
    evaluate_soak,
)


def test_extract_sink_publish_success_total_parses_success_counter() -> None:
    metrics = "\n".join(
        [
            "# HELP gateway_sink_publish_total Total data sink publish operations",
            "# TYPE gateway_sink_publish_total counter",
            'gateway_sink_publish_total{sink="redis_streams",topic="heber:events",status="success"} 42',
        ]
    )
    assert _extract_sink_publish_success_total(metrics) == 42.0


def test_evaluate_soak_passes_when_stream_grows_without_dlq_growth() -> None:
    config = SoakConfig(duration_seconds=120, poll_interval_seconds=30, max_allowed_dlq_growth=0)
    base = datetime(2026, 2, 12, 13, 30, tzinfo=UTC)
    snapshots = [
        PollSnapshot(
            ts=base,
            gateway_ready=True,
            sink_ready=True,
            sink_publish_total=10.0,
            stream_len=100,
            dlq_len=0,
            bronze_fresh=True,
            silver_fresh=True,
            errors=[],
        ),
        PollSnapshot(
            ts=base,
            gateway_ready=True,
            sink_ready=True,
            sink_publish_total=20.0,
            stream_len=130,
            dlq_len=0,
            bronze_fresh=True,
            silver_fresh=True,
            errors=[],
        ),
    ]

    passed, reasons, metrics = evaluate_soak(config, snapshots)

    assert passed is True
    assert reasons == []
    assert metrics["sink_publish_delta"] == 10.0
    assert metrics["stream_len_delta"] == 30
    assert metrics["dlq_growth"] == 0


def test_evaluate_soak_passes_when_sink_counter_flat_but_stream_and_files_are_healthy() -> None:
    config = SoakConfig(duration_seconds=120, poll_interval_seconds=30, max_allowed_dlq_growth=0)
    base = datetime(2026, 2, 12, 13, 30, tzinfo=UTC)
    snapshots = [
        PollSnapshot(
            ts=base,
            gateway_ready=True,
            sink_ready=True,
            sink_publish_total=10.0,
            stream_len=100,
            dlq_len=0,
            bronze_fresh=True,
            silver_fresh=True,
            errors=[],
        ),
        PollSnapshot(
            ts=base,
            gateway_ready=True,
            sink_ready=True,
            sink_publish_total=10.0,
            stream_len=110,
            dlq_len=0,
            bronze_fresh=True,
            silver_fresh=True,
            errors=[],
        ),
    ]

    passed, reasons, metrics = evaluate_soak(config, snapshots)

    assert passed is True
    assert reasons == []
    assert metrics["sink_publish_delta"] == 0.0
    assert metrics["stream_len_delta"] == 10


def test_default_gateway_key_uses_explicit_then_gw_api_keys_then_local_default(monkeypatch) -> None:
    monkeypatch.setenv("CERBERUS_GATEWAY_KEY", "gw_explicit")
    monkeypatch.setenv("GW_API_KEYS", "gw_list_1,gw_list_2")
    assert _default_gateway_key("http://localhost:8080") == "gw_explicit"

    monkeypatch.delenv("CERBERUS_GATEWAY_KEY", raising=False)
    assert _default_gateway_key("http://localhost:8080") == "gw_list_1"

    monkeypatch.delenv("GW_API_KEYS", raising=False)
    assert _default_gateway_key("http://localhost:8080") == "gw_cerberus_dev_key_12345"
    assert _default_gateway_key("https://gateway.prod.example") == ""


def test_evaluate_soak_fails_when_dlq_grows() -> None:
    config = SoakConfig(duration_seconds=120, poll_interval_seconds=30, max_allowed_dlq_growth=0)
    base = datetime(2026, 2, 12, 13, 30, tzinfo=UTC)
    snapshots = [
        PollSnapshot(
            ts=base,
            gateway_ready=True,
            sink_ready=True,
            sink_publish_total=10.0,
            stream_len=100,
            dlq_len=0,
            bronze_fresh=True,
            silver_fresh=True,
            errors=[],
        ),
        PollSnapshot(
            ts=base,
            gateway_ready=True,
            sink_ready=True,
            sink_publish_total=25.0,
            stream_len=140,
            dlq_len=3,
            bronze_fresh=True,
            silver_fresh=True,
            errors=[],
        ),
    ]

    passed, reasons, metrics = evaluate_soak(config, snapshots)

    assert passed is False
    assert any("DLQ growth" in reason for reason in reasons)
    assert metrics["dlq_growth"] == 3


def test_evaluate_soak_fails_when_gateway_readiness_drops() -> None:
    config = SoakConfig(duration_seconds=120, poll_interval_seconds=30, max_allowed_dlq_growth=0)
    base = datetime(2026, 2, 12, 13, 30, tzinfo=UTC)
    snapshots = [
        PollSnapshot(
            ts=base,
            gateway_ready=True,
            sink_ready=True,
            sink_publish_total=10.0,
            stream_len=100,
            dlq_len=0,
            bronze_fresh=True,
            silver_fresh=True,
            errors=[],
        ),
        PollSnapshot(
            ts=base,
            gateway_ready=False,
            sink_ready=False,
            sink_publish_total=11.0,
            stream_len=101,
            dlq_len=0,
            bronze_fresh=True,
            silver_fresh=True,
            errors=["gateway readiness failed"],
        ),
    ]

    passed, reasons, _ = evaluate_soak(config, snapshots)

    assert passed is False
    assert any("Gateway readiness failures" in reason for reason in reasons)


def test_collect_snapshot_captures_probe_timeout_without_crashing(monkeypatch) -> None:
    class _DummyClient:
        def get(self, path: str, **kwargs):
            if path == "/health/ready":
                return _response(
                    200,
                    {"status": "ready", "checks": {"sinks": "ok"}},
                )
            if path == "/metrics":
                return _response(
                    200,
                    text=('gateway_sink_publish_total{sink="redis_streams",topic="heber:events",status="success"} 5'),
                )
            raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(
        "scripts.soak_gateway_heber_monitor._probe_gateway_for_sink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    monkeypatch.setattr(
        "scripts.soak_gateway_heber_monitor._read_stream_len",
        lambda *_args, **_kwargs: (100, None),
    )
    monkeypatch.setattr(
        "scripts.soak_gateway_heber_monitor._latest_file_age_seconds",
        lambda *_args, **_kwargs: 0.0,
    )

    snapshot = _collect_snapshot(SoakConfig(), _DummyClient())

    assert snapshot.gateway_ready is True
    assert snapshot.sink_ready is True
    assert snapshot.errors
    assert any("probe request failed" in err for err in snapshot.errors)


def _response(status_code: int, payload: dict | None = None, text: str = ""):
    import httpx

    if payload is not None:
        return httpx.Response(status_code, json=payload)
    return httpx.Response(status_code, text=text)
