#!/usr/bin/env python3
"""Run a timed Gateway+Heber soak monitor and emit a markdown report."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def _first_csv_token(raw: str) -> str:
    for token in raw.split(","):
        candidate = token.strip()
        if candidate:
            return candidate
    return ""


def _default_gateway_key(gateway_url: str) -> str:
    explicit_key = os.getenv("CERBERUS_GATEWAY_KEY", "").strip()
    if explicit_key:
        return explicit_key

    gw_api_keys = _first_csv_token(os.getenv("GW_API_KEYS", ""))
    if gw_api_keys:
        return gw_api_keys

    normalized_url = gateway_url.strip().rstrip("/")
    if normalized_url in {"http://localhost:8080", "http://127.0.0.1:8080"}:
        return "gw_cerberus_dev_key_12345"

    return ""


def _extract_sink_publish_success_total(metrics_text: str) -> float | None:
    """Parse sink publish success counter from Gateway Prometheus metrics."""
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


@dataclass(frozen=True)
class SoakConfig:
    duration_seconds: int = 900
    poll_interval_seconds: int = 30
    gateway_url: str = "http://localhost:8080"
    gateway_key: str = ""
    timeout_seconds: float = 15.0
    redis_url: str = "redis://localhost:6379/0"
    stream_name: str = "heber:events"
    dlq_stream_name: str = "heber:events:dlq"
    heber_data_root: Path = Path("/Volumes/heber/data")
    required_dataset: str = "bars"
    freshness_max_age_seconds: float = 180.0
    max_allowed_dlq_growth: int = 0
    smoke_symbol: str = "AAPL"
    report_path: Path = Path("docs/audits/gateway-heber-soak-report.md")

    @classmethod
    def from_env(cls, args: argparse.Namespace) -> "SoakConfig":
        duration_seconds = int(
            getattr(args, "duration_seconds", 0) or os.getenv("CERBERUS_SOAK_DURATION_SECONDS", "900")
        )
        poll_interval_seconds = int(
            getattr(args, "poll_interval_seconds", 0) or os.getenv("CERBERUS_SOAK_POLL_INTERVAL_SECONDS", "30")
        )
        report_path_raw = str(
            getattr(args, "report_path", "")
            or os.getenv("CERBERUS_SOAK_REPORT_PATH", "docs/audits/gateway-heber-soak-report.md")
        )
        gateway_url = os.getenv("CERBERUS_GATEWAY_URL", "http://localhost:8080")
        return cls(
            duration_seconds=max(30, duration_seconds),
            poll_interval_seconds=max(5, poll_interval_seconds),
            gateway_url=gateway_url,
            gateway_key=_default_gateway_key(gateway_url),
            timeout_seconds=float(os.getenv("CERBERUS_SOAK_TIMEOUT_SECONDS", "15")),
            redis_url=os.getenv("CERBERUS_SOAK_REDIS_URL", "redis://localhost:6379/0"),
            stream_name=os.getenv("HEBER_REDIS_STREAM_NAME", "heber:events"),
            dlq_stream_name=os.getenv("HEBER_REDIS_DLQ_STREAM_NAME", "heber:events:dlq"),
            heber_data_root=Path(
                os.getenv(
                    "CERBERUS_HEBER_DATA_ROOT",
                    os.getenv("HEBER_DATA_ROOT", "/Volumes/heber/data"),
                )
            ),
            required_dataset=os.getenv("CERBERUS_SOAK_REQUIRED_DATASET", "bars"),
            freshness_max_age_seconds=float(os.getenv("CERBERUS_SOAK_FRESHNESS_MAX_AGE_SECONDS", "180")),
            max_allowed_dlq_growth=int(os.getenv("CERBERUS_SOAK_MAX_ALLOWED_DLQ_GROWTH", "0")),
            smoke_symbol=os.getenv("CERBERUS_SMOKE_SYMBOL", "AAPL").upper(),
            report_path=Path(report_path_raw),
        )


@dataclass(frozen=True)
class PollSnapshot:
    ts: datetime
    gateway_ready: bool
    sink_ready: bool
    sink_publish_total: float | None
    stream_len: int | None
    dlq_len: int | None
    bronze_fresh: bool
    silver_fresh: bool
    errors: list[str]


def _run_redis_cli(redis_url: str, *args: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["redis-cli", "-u", redis_url, *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "redis-cli not found in PATH"

    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip() or "redis-cli error"
        return False, error_text
    return True, completed.stdout.strip()


def _read_stream_len(redis_url: str, stream_name: str) -> tuple[int | None, str | None]:
    ok, output = _run_redis_cli(redis_url, "XLEN", stream_name)
    if not ok:
        return None, output
    try:
        return int(output), None
    except ValueError:
        return None, f"unexpected XLEN output for {stream_name}: {output}"


def _latest_file_age_seconds(data_root: Path, layer: str, dataset: str) -> float | None:
    if layer == "bronze":
        roots = [data_root / "bronze" / f"feed={dataset}"]
        roots.extend(sorted((data_root / "bronze").glob(f"provider=*/feed={dataset}")))
        suffix = "*.jsonl.gz"
    else:
        roots = [data_root / "silver" / f"feed={dataset}"]
        suffix = "*.parquet"

    latest_path: Path | None = None
    latest_mtime = -1.0
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob(suffix):
            if not candidate.is_file():
                continue
            mtime = candidate.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = candidate

    if latest_path is None:
        return None
    return max(0.0, time.time() - latest_mtime)


def _probe_gateway_for_sink(config: SoakConfig, client: httpx.Client) -> tuple[bool, str]:
    headers: dict[str, str] = {"X-Gateway-Cache": "bypass"}
    if config.gateway_key:
        headers["X-Gateway-Key"] = config.gateway_key

    response = client.get(
        f"/api/v1/alpaca/stocks/{config.smoke_symbol}/bars",
        params={"timeframe": "1Min", "limit": 1, "cache_buster": str(time.time_ns())},
        headers=headers,
    )
    if response.status_code != 200:
        return False, f"gateway probe returned HTTP {response.status_code}"
    return True, "ok"


def _collect_snapshot(config: SoakConfig, client: httpx.Client) -> PollSnapshot:
    errors: list[str] = []
    gateway_ready = False
    sink_ready = False

    try:
        ok, detail = _probe_gateway_for_sink(config, client)
        if not ok:
            errors.append(detail)
    except Exception as exc:
        errors.append(f"gateway probe request failed: {type(exc).__name__}: {exc}")

    try:
        ready_response = client.get("/health/ready")
        if ready_response.status_code == 200:
            payload_data = ready_response.json()
            payload = payload_data if isinstance(payload_data, dict) else {}
            gateway_ready = payload.get("status") in {"ready", "ok"}
            checks = payload.get("checks")
            sink_ready = isinstance(checks, dict) and checks.get("sinks") == "ok"
        else:
            errors.append(f"gateway readiness HTTP {ready_response.status_code}")
    except Exception as exc:
        errors.append(f"gateway readiness request failed: {type(exc).__name__}: {exc}")

    sink_publish_total: float | None = None
    try:
        metrics_response = client.get(
            "/metrics",
            headers={"X-Gateway-Key": config.gateway_key} if config.gateway_key else None,
        )
        if metrics_response.status_code == 200:
            sink_publish_total = _extract_sink_publish_success_total(metrics_response.text)
            if sink_publish_total is None:
                errors.append("missing gateway sink publish success metric")
        else:
            errors.append(f"gateway metrics HTTP {metrics_response.status_code}")
    except Exception as exc:
        errors.append(f"gateway metrics request failed: {type(exc).__name__}: {exc}")

    stream_len, stream_error = _read_stream_len(config.redis_url, config.stream_name)
    if stream_error:
        errors.append(stream_error)
    dlq_len, dlq_error = _read_stream_len(config.redis_url, config.dlq_stream_name)
    if dlq_error:
        errors.append(dlq_error)

    bronze_age = _latest_file_age_seconds(config.heber_data_root, "bronze", config.required_dataset)
    silver_age = _latest_file_age_seconds(config.heber_data_root, "silver", config.required_dataset)

    bronze_fresh = bronze_age is not None and bronze_age <= config.freshness_max_age_seconds
    silver_fresh = silver_age is not None and silver_age <= config.freshness_max_age_seconds

    return PollSnapshot(
        ts=datetime.now(UTC),
        gateway_ready=gateway_ready,
        sink_ready=sink_ready,
        sink_publish_total=sink_publish_total,
        stream_len=stream_len,
        dlq_len=dlq_len,
        bronze_fresh=bronze_fresh,
        silver_fresh=silver_fresh,
        errors=errors,
    )


def evaluate_soak(config: SoakConfig, snapshots: list[PollSnapshot]) -> tuple[bool, list[str], dict[str, Any]]:
    if not snapshots:
        return False, ["No snapshots collected during soak"], {}

    first = snapshots[0]
    last = snapshots[-1]

    gateway_failures = sum(1 for snap in snapshots if not snap.gateway_ready or not snap.sink_ready)
    sink_publish_delta = 0.0
    if first.sink_publish_total is not None and last.sink_publish_total is not None:
        sink_publish_delta = last.sink_publish_total - first.sink_publish_total

    stream_len_delta = 0
    if first.stream_len is not None and last.stream_len is not None:
        stream_len_delta = last.stream_len - first.stream_len

    dlq_growth = 0
    if first.dlq_len is not None and last.dlq_len is not None:
        dlq_growth = last.dlq_len - first.dlq_len

    bronze_fresh_seen = any(snap.bronze_fresh for snap in snapshots)
    silver_fresh_seen = any(snap.silver_fresh for snap in snapshots)
    error_snapshots = sum(1 for snap in snapshots if snap.errors)

    reasons: list[str] = []
    if gateway_failures > 0:
        reasons.append(f"Gateway readiness failures observed in {gateway_failures} poll(s)")
    if stream_len_delta <= 0:
        reasons.append("Redis stream length did not grow")
    if dlq_growth > config.max_allowed_dlq_growth:
        reasons.append(f"DLQ growth exceeded threshold ({dlq_growth} > {config.max_allowed_dlq_growth})")
    if not bronze_fresh_seen:
        reasons.append("No fresh Bronze write observed during soak window")
    if not silver_fresh_seen:
        reasons.append("No fresh Silver write observed during soak window")
    if error_snapshots > 0:
        reasons.append(f"Polling errors observed in {error_snapshots} snapshot(s)")

    metrics: dict[str, Any] = {
        "poll_count": len(snapshots),
        "gateway_failures": gateway_failures,
        "sink_publish_delta": sink_publish_delta,
        "stream_len_delta": stream_len_delta,
        "dlq_growth": dlq_growth,
        "bronze_fresh_seen": bronze_fresh_seen,
        "silver_fresh_seen": silver_fresh_seen,
        "error_snapshots": error_snapshots,
        "start_ts": first.ts.isoformat(),
        "end_ts": last.ts.isoformat(),
    }
    return len(reasons) == 0, reasons, metrics


def _render_report(
    config: SoakConfig, snapshots: list[PollSnapshot], passed: bool, reasons: list[str], metrics: dict[str, Any]
) -> str:
    status = "PASS" if passed else "FAIL"
    lines = [
        "# Gateway + Heber Soak Report",
        "",
        f"- Run date (UTC): `{datetime.now(UTC).isoformat()}`",
        f"- Status: **{status}**",
        f"- Duration seconds: `{config.duration_seconds}`",
        f"- Poll interval seconds: `{config.poll_interval_seconds}`",
        f"- Gateway URL: `{config.gateway_url}`",
        f"- Redis URL: `{config.redis_url}`",
        f"- Stream: `{config.stream_name}`",
        f"- DLQ stream: `{config.dlq_stream_name}`",
        "",
        "## Summary Metrics",
        "",
        f"- Poll count: `{metrics.get('poll_count', 0)}`",
        f"- Gateway readiness failures: `{metrics.get('gateway_failures', 0)}`",
        f"- Sink publish delta: `{metrics.get('sink_publish_delta', 0)}`",
        f"- Stream length delta: `{metrics.get('stream_len_delta', 0)}`",
        f"- DLQ growth: `{metrics.get('dlq_growth', 0)}`",
        f"- Bronze fresh seen: `{metrics.get('bronze_fresh_seen', False)}`",
        f"- Silver fresh seen: `{metrics.get('silver_fresh_seen', False)}`",
        f"- Poll error snapshots: `{metrics.get('error_snapshots', 0)}`",
        "",
        "## Verdict",
        "",
    ]
    if reasons:
        lines.append("- Blocking issues:")
        lines.extend([f"  - {reason}" for reason in reasons])
    else:
        lines.append("- No blocking issues observed in this soak window.")

    lines.extend(
        [
            "",
            "## Poll Timeline",
            "",
            "| UTC Timestamp | Gateway Ready | Sink Ready | Sink Counter | Stream Len | DLQ Len | Bronze Fresh | Silver Fresh | Errors |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for snap in snapshots:
        lines.append(
            "| "
            f"{snap.ts.isoformat()} | "
            f"{'yes' if snap.gateway_ready else 'no'} | "
            f"{'yes' if snap.sink_ready else 'no'} | "
            f"{snap.sink_publish_total if snap.sink_publish_total is not None else 'n/a'} | "
            f"{snap.stream_len if snap.stream_len is not None else 'n/a'} | "
            f"{snap.dlq_len if snap.dlq_len is not None else 'n/a'} | "
            f"{'yes' if snap.bronze_fresh else 'no'} | "
            f"{'yes' if snap.silver_fresh else 'no'} | "
            f"{'; '.join(snap.errors) if snap.errors else ''} |"
        )

    return "\n".join(lines) + "\n"


def run_soak(config: SoakConfig) -> int:
    snapshots: list[PollSnapshot] = []
    deadline = time.time() + float(config.duration_seconds)

    with httpx.Client(base_url=config.gateway_url, timeout=config.timeout_seconds) as client:
        while True:
            snapshot = _collect_snapshot(config, client)
            snapshots.append(snapshot)
            print(
                f"[{snapshot.ts.isoformat()}] "
                f"gateway_ready={snapshot.gateway_ready} "
                f"sink_ready={snapshot.sink_ready} "
                f"sink_total={snapshot.sink_publish_total} "
                f"stream={snapshot.stream_len} "
                f"dlq={snapshot.dlq_len} "
                f"bronze_fresh={snapshot.bronze_fresh} "
                f"silver_fresh={snapshot.silver_fresh} "
                f"errors={len(snapshot.errors)}"
            )
            if time.time() >= deadline:
                break
            time.sleep(float(config.poll_interval_seconds))

    passed, reasons, metrics = evaluate_soak(config, snapshots)
    report = _render_report(config, snapshots, passed, reasons, metrics)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(report, encoding="utf-8")

    print(f"\nSoak report written: {config.report_path}")
    print(f"Soak status: {'PASS' if passed else 'FAIL'}")
    if reasons:
        for reason in reasons:
            print(f"- {reason}")

    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gateway+Heber soak monitor and write report")
    parser.add_argument("--duration-seconds", type=int, default=0, help="Soak duration in seconds")
    parser.add_argument("--poll-interval-seconds", type=int, default=0, help="Polling interval in seconds")
    parser.add_argument("--report-path", default="", help="Report output path")
    args = parser.parse_args()

    config = SoakConfig.from_env(args)
    return run_soak(config)


if __name__ == "__main__":
    raise SystemExit(main())
