#!/usr/bin/env python3
"""One-command smoke checks for Cerberus -> Data-Gateway -> Heber integration."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _datasets_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/api/v1"):
        return f"{normalized}/datasets"
    if normalized.endswith("/datasets"):
        return normalized
    return f"{normalized}/api/v1/datasets"


@dataclass(frozen=True)
class SmokeConfig:
    gateway_url: str
    gateway_key: str
    heber_catalog_url: str
    heber_data_root: Path
    smoke_symbol: str
    timeout_seconds: float
    require_sink: bool
    require_silver_file: bool
    required_dataset: str
    sink_metric_poll_attempts: int
    sink_metric_poll_interval_seconds: float
    write_poll_timeout_seconds: float
    write_poll_interval_seconds: float

    @classmethod
    def from_env(cls) -> "SmokeConfig":
        return cls(
            gateway_url=os.getenv("CERBERUS_GATEWAY_URL", "http://localhost:8080"),
            gateway_key=os.getenv("CERBERUS_GATEWAY_KEY", ""),
            heber_catalog_url=os.getenv(
                "CERBERUS_HEBER_CATALOG_URL",
                "http://localhost:8085/api/v1",
            ),
            heber_data_root=Path(
                os.getenv(
                    "CERBERUS_HEBER_DATA_ROOT",
                    os.getenv("HEBER_DATA_ROOT", "/Volumes/heber/data"),
                )
            ),
            smoke_symbol=os.getenv("CERBERUS_SMOKE_SYMBOL", "AAPL").upper(),
            timeout_seconds=float(os.getenv("CERBERUS_SMOKE_TIMEOUT_SECONDS", "5")),
            require_sink=_env_bool("CERBERUS_SMOKE_REQUIRE_SINK", True),
            require_silver_file=_env_bool("CERBERUS_SMOKE_REQUIRE_SILVER_FILE", True),
            required_dataset=os.getenv("CERBERUS_SMOKE_REQUIRED_DATASET", "bars"),
            sink_metric_poll_attempts=max(1, int(os.getenv("CERBERUS_SMOKE_SINK_METRIC_POLL_ATTEMPTS", "5"))),
            sink_metric_poll_interval_seconds=float(os.getenv("CERBERUS_SMOKE_SINK_METRIC_POLL_INTERVAL_SECONDS", "1")),
            write_poll_timeout_seconds=float(os.getenv("CERBERUS_SMOKE_WRITE_POLL_TIMEOUT_SECONDS", "45")),
            write_poll_interval_seconds=float(os.getenv("CERBERUS_SMOKE_WRITE_POLL_INTERVAL_SECONDS", "2")),
        )


def check_gateway_authenticated(
    *,
    client: httpx.Client,
    gateway_key: str,
) -> tuple[bool, str]:
    """Validate Cerberus -> Gateway authenticated API call."""
    headers = {"X-Gateway-Key": gateway_key} if gateway_key else None
    response = client.get(
        "/api/v1/alpaca/screener/most-actives",
        params={"by": "volume", "top": 1},
        headers=headers,
    )
    if response.status_code != 200:
        return False, f"gateway auth call failed with HTTP {response.status_code}"
    return True, "gateway authenticated call ok"


def check_gateway_sink_ready(
    *,
    client: httpx.Client,
    require_sink: bool,
) -> tuple[bool, str]:
    """Validate Gateway readiness and sink status."""
    response = client.get("/health/ready")
    if response.status_code != 200:
        return False, f"gateway readiness failed with HTTP {response.status_code}"

    payload: dict[str, Any] = {}
    parsed = response.json()
    if isinstance(parsed, dict):
        payload = parsed

    if payload.get("status") not in {"ready", "ok"}:
        return False, f"gateway readiness returned status={payload.get('status')}"

    checks = payload.get("checks")
    if require_sink:
        if not isinstance(checks, dict) or checks.get("sinks") != "ok":
            return False, "gateway readiness did not report sinks=ok"
    return True, "gateway readiness and sink checks ok"


def _extract_sink_publish_success_total(metrics_text: str) -> float | None:
    """Parse `gateway_sink_publish_total` counter for Redis stream success publishes."""
    for line in metrics_text.splitlines():
        if not line.startswith("gateway_sink_publish_total{"):
            continue
        if 'sink="redis_streams"' not in line:
            continue
        if 'topic="heber:events"' not in line:
            continue
        if 'status="success"' not in line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            return float(parts[1])
        except ValueError:
            continue
    return None


def _read_sink_publish_success_total(
    *,
    client: httpx.Client,
    gateway_key: str,
) -> tuple[bool, float, str]:
    """Read Redis stream publish success counter from Gateway Prometheus metrics."""
    headers = {"X-Gateway-Key": gateway_key} if gateway_key else None
    response = client.get("/metrics", headers=headers)
    if response.status_code != 200:
        return False, 0.0, f"gateway metrics failed with HTTP {response.status_code}"

    value = _extract_sink_publish_success_total(response.text)
    if value is None:
        return False, 0.0, "gateway metrics missing sink publish success counter"

    return True, value, "gateway sink publish counter read"


def check_gateway_sink_publish_activity(
    *,
    client: httpx.Client,
    gateway_key: str,
    baseline_success_count: float,
    min_increment: float = 1.0,
    poll_attempts: int = 5,
    poll_interval_seconds: float = 1.0,
) -> tuple[bool, str]:
    """Validate sink publish success counter increased after smoke probe request."""
    last_value = baseline_success_count
    attempts = max(1, int(poll_attempts))
    for attempt in range(1, attempts + 1):
        ok, current, detail = _read_sink_publish_success_total(
            client=client,
            gateway_key=gateway_key,
        )
        if not ok:
            return False, detail

        last_value = current
        delta = current - baseline_success_count
        if delta >= min_increment:
            return True, f"sink publish activity observed (delta={delta:.1f}, total={current:.1f})"

        if attempt < attempts and poll_interval_seconds > 0:
            time.sleep(poll_interval_seconds)

    return (
        False,
        (
            "gateway sink publish counter did not increase enough "
            f"(baseline={baseline_success_count:.1f}, current={last_value:.1f}, required_delta={min_increment:.1f})"
        ),
    )


def check_heber_catalog(catalog_url: str, timeout_seconds: float) -> tuple[bool, str]:
    """Validate Heber catalog connectivity."""
    datasets_url = _datasets_url(catalog_url)
    response = httpx.get(datasets_url, timeout=timeout_seconds)
    if response.status_code != 200:
        return False, f"heber catalog datasets call failed with HTTP {response.status_code}"
    return True, "heber catalog datasets endpoint ok"


def check_heber_silver_partition(config: SmokeConfig) -> tuple[bool, str]:
    """Validate that Heber Silver has at least one partition file."""
    dataset_root = config.heber_data_root / "silver" / f"feed={config.required_dataset}"
    if not dataset_root.exists():
        return False, f"missing Silver dataset path: {dataset_root}"

    sample_file = next(dataset_root.rglob("*.parquet"), None)
    if sample_file is None:
        return False, f"no parquet files found under {dataset_root}"
    if sample_file.stat().st_size <= 0:
        return False, f"empty parquet file detected: {sample_file}"

    return True, f"Silver parquet found: {sample_file}"


def check_heber_layer_has_fresh_file(
    *,
    config: SmokeConfig,
    layer: str,
    baseline_time: datetime,
    poll_timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[bool, str]:
    """Validate Bronze/Silver has a file newer than the probe start time."""
    if layer not in {"bronze", "silver"}:
        return False, f"unsupported layer: {layer}"

    suffix = "*.jsonl.gz" if layer == "bronze" else "*.parquet"
    dataset_root = config.heber_data_root / layer / f"feed={config.required_dataset}"
    if not dataset_root.exists():
        return False, f"missing {layer} dataset path: {dataset_root}"

    deadline = time.time() + max(0.0, poll_timeout_seconds)
    baseline_epoch = baseline_time.timestamp()
    latest_file: Path | None = None

    while True:
        candidates = [p for p in dataset_root.rglob(suffix) if p.is_file()]
        if candidates:
            latest_file = max(candidates, key=lambda p: p.stat().st_mtime)
            if latest_file.stat().st_mtime >= baseline_epoch:
                return True, f"{layer} fresh file found: {latest_file}"

        if time.time() >= deadline:
            break
        if poll_interval_seconds > 0:
            time.sleep(poll_interval_seconds)

    if latest_file is None:
        return False, f"no {suffix} files found under {dataset_root}"

    last_mtime = datetime.fromtimestamp(latest_file.stat().st_mtime, tz=UTC).isoformat()
    return False, f"{layer} latest file is stale (mtime={last_mtime}): {latest_file}"


def run_smoke(config: SmokeConfig) -> int:
    """Run all smoke checks and return process exit code."""
    results: list[tuple[str, bool, str]] = []
    smoke_started_at = datetime.now(UTC)
    sink_ready_ok = False
    bronze_fresh_ok: bool | None = None
    silver_fresh_ok: bool | None = None

    with httpx.Client(base_url=config.gateway_url, timeout=config.timeout_seconds) as gateway_client:
        sink_baseline = 0.0
        if config.require_sink:
            ok, sink_baseline, detail = _read_sink_publish_success_total(
                client=gateway_client,
                gateway_key=config.gateway_key,
            )
            results.append(("Gateway sink metrics readable", ok, detail))

        ok, detail = check_gateway_authenticated(
            client=gateway_client,
            gateway_key=config.gateway_key,
        )
        results.append(("Cerberus -> Gateway authenticated call", ok, detail))

        ok, detail = check_gateway_sink_ready(
            client=gateway_client,
            require_sink=config.require_sink,
        )
        sink_ready_ok = ok
        results.append(("Gateway -> Redis sink readiness", ok, detail))

        if config.require_sink:
            ok, detail = check_gateway_sink_publish_activity(
                client=gateway_client,
                gateway_key=config.gateway_key,
                baseline_success_count=sink_baseline,
                min_increment=1.0,
                poll_attempts=config.sink_metric_poll_attempts,
                poll_interval_seconds=config.sink_metric_poll_interval_seconds,
            )
            results.append(("Gateway -> Redis stream publish activity", ok, detail))

    ok, detail = check_heber_catalog(config.heber_catalog_url, config.timeout_seconds)
    results.append(("Heber catalog datasets endpoint", ok, detail))

    if config.require_silver_file:
        ok, detail = check_heber_silver_partition(config)
    else:
        ok, detail = True, "Silver file check skipped"
    results.append(("Heber Silver partition presence", ok, detail))

    if config.require_silver_file:
        ok, detail = check_heber_layer_has_fresh_file(
            config=config,
            layer="bronze",
            baseline_time=smoke_started_at,
            poll_timeout_seconds=config.write_poll_timeout_seconds,
            poll_interval_seconds=config.write_poll_interval_seconds,
        )
        bronze_fresh_ok = ok
        results.append(("Heber Bronze fresh write", ok, detail))

        ok, detail = check_heber_layer_has_fresh_file(
            config=config,
            layer="silver",
            baseline_time=smoke_started_at,
            poll_timeout_seconds=config.write_poll_timeout_seconds,
            poll_interval_seconds=config.write_poll_interval_seconds,
        )
        silver_fresh_ok = ok
        results.append(("Heber Silver fresh write", ok, detail))

    if config.require_sink and sink_ready_ok and (bronze_fresh_ok is False or silver_fresh_ok is False):
        results.append(
            (
                "Gateway sink -> Heber write propagation",
                False,
                (
                    "Gateway sink is ready but no recent Heber writes were observed. "
                    "Check Heber consumer/watcher health and ingest contract mappings."
                ),
            )
        )

    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} - {detail}")
        if not ok:
            failed += 1

    if failed:
        print(f"\nSmoke check failed ({failed}/{len(results)} checks failed).")
        return 1

    print(f"\nSmoke check passed ({len(results)}/{len(results)} checks passed).")
    return 0


def main() -> int:
    config = SmokeConfig.from_env()
    return run_smoke(config)


if __name__ == "__main__":
    raise SystemExit(main())
