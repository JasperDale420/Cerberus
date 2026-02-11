#!/usr/bin/env python3
"""One-command smoke checks for Cerberus -> Data-Gateway -> Heber integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
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


def run_smoke(config: SmokeConfig) -> int:
    """Run all smoke checks and return process exit code."""
    results: list[tuple[str, bool, str]] = []

    with httpx.Client(base_url=config.gateway_url, timeout=config.timeout_seconds) as gateway_client:
        ok, detail = check_gateway_authenticated(
            client=gateway_client,
            gateway_key=config.gateway_key,
        )
        results.append(("Cerberus -> Gateway authenticated call", ok, detail))

        ok, detail = check_gateway_sink_ready(
            client=gateway_client,
            require_sink=config.require_sink,
        )
        results.append(("Gateway -> Redis sink readiness", ok, detail))

    ok, detail = check_heber_catalog(config.heber_catalog_url, config.timeout_seconds)
    results.append(("Heber catalog datasets endpoint", ok, detail))

    if config.require_silver_file:
        ok, detail = check_heber_silver_partition(config)
    else:
        ok, detail = True, "Silver file check skipped"
    results.append(("Heber Silver partition presence", ok, detail))

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
