"""
Healthcheck module for Cerberus trading system.

Provides system health verification for monitoring and operational readiness.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import httpx

from src.core.settings import get_settings

VERSION = "1.0.0"


def check_database_connectivity(db_path: str = "cerberus.db") -> Dict[str, Any]:
    """Verify SQLite database is accessible and not corrupted."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()

        # Test basic query
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        cursor.fetchone()

        # Check integrity
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        integrity_ok = result and result[0] == "ok"
        return {
            "status": "ok" if integrity_ok else "degraded",
            "path": db_path,
            "exists": True,
            "integrity": "ok" if integrity_ok else "check_failed",
        }
    except sqlite3.OperationalError as e:
        return {
            "status": "error",
            "path": db_path,
            "exists": Path(db_path).exists(),
            "error": str(e),
        }
    except Exception as e:
        return {
            "status": "error",
            "path": db_path,
            "error": f"Unexpected error: {type(e).__name__}: {e}",
        }


def check_alpaca_credentials() -> Dict[str, Any]:
    """Verify Alpaca API credentials are configured."""
    settings = get_settings()
    api_key = settings.resolved_api_key
    secret_key = settings.resolved_secret_key
    base_url = settings.resolved_base_url

    if not api_key or not secret_key:
        return {
            "status": "skipped",
            "reason": "Alpaca credentials not configured (not required for gateway mode)",
        }

    is_paper = settings.alpaca_paper
    mode = "paper" if is_paper else "live"

    return {
        "status": "ok",
        "mode": mode,
        "base_url": base_url or "not_set",
        "credentials_present": True,
    }


def check_gateway_connectivity() -> Dict[str, Any]:
    """Verify Data-Gateway is reachable."""
    settings = get_settings()
    gateway_url = str(settings.cerberus_gateway_url).strip()
    if not gateway_url:
        return {
            "status": "skipped",
            "reason": "CERBERUS_GATEWAY_URL not set",
        }

    headers: Dict[str, str] = {}
    if settings.cerberus_gateway_key:
        headers["X-Gateway-Key"] = settings.cerberus_gateway_key

    check_url = f"{gateway_url.rstrip('/')}/health/ready"
    try:
        response = httpx.get(
            check_url,
            headers=headers or None,
            timeout=max(1.0, float(settings.cerberus_gateway_timeout_seconds)),
        )
        payload: Dict[str, Any] = {}
        try:
            payload_data = response.json()
            if isinstance(payload_data, dict):
                payload = payload_data
        except Exception:
            payload = {}

        ok = response.status_code == 200 and payload.get("status") in {"ready", "ok"}
        return {
            "status": "ok" if ok else "degraded",
            "url": check_url,
            "http_status": response.status_code,
            "response_status": payload.get("status"),
        }
    except Exception as e:
        return {
            "status": "error",
            "url": check_url,
            "error": f"{type(e).__name__}: {e}",
        }


def check_heber_connectivity() -> Dict[str, Any]:
    """Verify Heber catalog endpoint is reachable when configured."""
    settings = get_settings()
    catalog_url = str(settings.cerberus_heber_catalog_url).strip()
    if not catalog_url:
        return {
            "status": "skipped",
            "reason": "CERBERUS_HEBER_CATALOG_URL not set",
        }

    check_url = f"{catalog_url.rstrip('/')}/datasets"
    try:
        response = httpx.get(check_url, timeout=5.0)
        return {
            "status": "ok" if response.status_code == 200 else "degraded",
            "url": check_url,
            "http_status": response.status_code,
        }
    except Exception as e:
        return {
            "status": "error",
            "url": check_url,
            "error": f"{type(e).__name__}: {e}",
        }


def _latest_dataset_file(data_root: Path, feed: str) -> Path | None:
    """Find the most recent parquet file in the latest date partition.

    Uses a targeted scan of only the most recent partition directory
    rather than a full rglob, which can take 55+ seconds over Docker
    volume mounts on large data lakes.
    """
    dataset_root = data_root / "silver" / f"feed={feed}"
    if not dataset_root.exists():
        return None

    # Collect all date partition directories (e.g. dt=2026-03-24/).
    # The Silver schema includes an optional instrument_type= level between
    # feed= and dt=, so we search both direct children and one level deeper.
    raw_date_dirs: list[Path] = []
    for child in dataset_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("dt="):
            raw_date_dirs.append(child)
        else:
            # Descend one level (e.g. instrument_type=equity)
            for grandchild in child.iterdir():
                if grandchild.is_dir() and grandchild.name.startswith("dt="):
                    raw_date_dirs.append(grandchild)

    date_dirs = sorted(raw_date_dirs, key=lambda p: p.name, reverse=True)

    # Check the most recent date partitions for parquet files (up to 3)
    for date_dir in date_dirs[:3]:
        latest: Path | None = None
        latest_mtime = -1.0
        for parquet_file in date_dir.rglob("*.parquet"):
            if parquet_file.name.startswith("._"):
                continue
            if not parquet_file.is_file():
                continue
            mtime = parquet_file.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = parquet_file
        if latest is not None:
            return latest

    # Fallback: no date partitions found, try a shallow glob
    for parquet_file in dataset_root.glob("*.parquet"):
        if parquet_file.name.startswith("._"):
            continue
        if parquet_file.is_file():
            return parquet_file

    return None


def check_heber_freshness(
    required_feeds: tuple[str, ...] = ("bars", "trades"),
    max_age_seconds: float = 900.0,
) -> Dict[str, Any]:
    """Verify Heber Silver datasets contain recent files for required feeds."""
    settings = get_settings()
    data_root_raw = str(settings.cerberus_heber_data_root).strip()
    if not data_root_raw:
        return {
            "status": "skipped",
            "reason": "CERBERUS_HEBER_DATA_ROOT not set",
        }

    data_root = Path(data_root_raw)
    if not data_root.exists():
        return {
            "status": "error",
            "reason": f"Heber data root does not exist: {data_root}",
        }

    now_epoch = datetime.now(timezone.utc).timestamp()
    freshness: Dict[str, Dict[str, Any]] = {}
    stale_feeds: list[str] = []

    for feed in required_feeds:
        latest = _latest_dataset_file(data_root, feed)
        if latest is None:
            stale_feeds.append(feed)
            freshness[feed] = {
                "status": "missing",
                "file": None,
                "age_seconds": None,
            }
            continue

        age_seconds = max(0.0, now_epoch - latest.stat().st_mtime)
        feed_status = "ok" if age_seconds <= max_age_seconds else "stale"
        if feed_status != "ok":
            stale_feeds.append(feed)

        freshness[feed] = {
            "status": feed_status,
            "file": str(latest),
            "age_seconds": round(age_seconds, 3),
            "max_age_seconds": float(max_age_seconds),
        }

    return {
        "status": "ok" if not stale_feeds else "degraded",
        "data_root": str(data_root),
        "stale_feeds": stale_feeds,
        "freshness": freshness,
    }


def get_system_info() -> Dict[str, Any]:
    """Get basic system and version information."""
    import sys

    return {
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def run_healthcheck(verbose: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive healthcheck.

    Args:
        verbose: If True, print human-readable output to console

    Returns:
        Dictionary with health status of all components
    """
    results = {
        "system": get_system_info(),
        "database": check_database_connectivity(),
        "alpaca": check_alpaca_credentials(),
        "gateway": check_gateway_connectivity(),
        "heber": check_heber_connectivity(),
        "heber_freshness": check_heber_freshness(),
    }

    # Determine overall health
    all_ok = True
    for key, component in results.items():
        if key == "system" or not isinstance(component, dict):
            continue
        if component.get("status") not in {"ok", "skipped"}:
            all_ok = False
            break
    results_with_status: Dict[str, Any] = dict(results)
    results_with_status["overall_status"] = "ok" if all_ok else "degraded"

    if verbose:
        print_healthcheck_results(results_with_status)

    return results_with_status


def print_healthcheck_results(results: Dict[str, Any]) -> None:
    """Print human-readable healthcheck results."""
    print("\n" + "=" * 60)
    print("Cerberus Trading System - Healthcheck")
    print("=" * 60 + "\n")

    # System info
    sys_info = results.get("system", {})
    print(f"Version: {sys_info.get('version', 'unknown')}")
    print(f"Timestamp: {sys_info.get('timestamp', 'unknown')}")
    print(f"Python: {sys_info.get('python_version', 'unknown')}\n")

    # Database
    db = results.get("database", {})
    status_symbol = "✓" if db.get("status") == "ok" else "✗"
    print(f"{status_symbol} Database connectivity: {db.get('status', 'unknown').upper()}")
    if db.get("status") != "ok":
        print(f"  └─ Error: {db.get('error', 'unknown')}")
    else:
        print(f"  └─ Path: {db.get('path', 'unknown')}")
    print()

    # Alpaca
    alpaca = results.get("alpaca", {})
    status_symbol = "✓" if alpaca.get("status") in {"ok", "skipped"} else "✗"
    print(f"{status_symbol} Alpaca API: {alpaca.get('status', 'unknown').upper()}")
    if alpaca.get("status") not in {"ok", "skipped"}:
        print(f"  └─ Error: {alpaca.get('error', 'unknown')}")
    elif alpaca.get("status") == "skipped":
        print(f"  └─ Reason: {alpaca.get('reason', 'unknown')}")
    else:
        print(f"  └─ Mode: {alpaca.get('mode', 'unknown')}")
        print(f"  └─ Base URL: {alpaca.get('base_url', 'unknown')}")
    print()

    # Data-Gateway
    gateway = results.get("gateway", {})
    status_symbol = "✓" if gateway.get("status") in {"ok", "skipped"} else "✗"
    print(f"{status_symbol} Data-Gateway: {gateway.get('status', 'unknown').upper()}")
    if gateway.get("status") == "ok":
        print(f"  └─ URL: {gateway.get('url', 'unknown')}")
        print(f"  └─ HTTP: {gateway.get('http_status', 'unknown')}")
    elif gateway.get("status") == "skipped":
        print(f"  └─ Reason: {gateway.get('reason', 'unknown')}")
    else:
        print(f"  └─ Error: {gateway.get('error', 'unknown')}")
    print()

    # Heber
    heber = results.get("heber", {})
    status_symbol = "✓" if heber.get("status") in {"ok", "skipped"} else "✗"
    print(f"{status_symbol} Heber Catalog: {heber.get('status', 'unknown').upper()}")
    if heber.get("status") == "ok":
        print(f"  └─ URL: {heber.get('url', 'unknown')}")
        print(f"  └─ HTTP: {heber.get('http_status', 'unknown')}")
    elif heber.get("status") == "skipped":
        print(f"  └─ Reason: {heber.get('reason', 'unknown')}")
    else:
        print(f"  └─ Error: {heber.get('error', 'unknown')}")
    print()

    # Heber freshness
    heber_freshness = results.get("heber_freshness", {})
    status_symbol = "✓" if heber_freshness.get("status") in {"ok", "skipped"} else "✗"
    print(f"{status_symbol} Heber Freshness: {heber_freshness.get('status', 'unknown').upper()}")
    if heber_freshness.get("status") == "ok":
        print(f"  └─ Data Root: {heber_freshness.get('data_root', 'unknown')}")
    elif heber_freshness.get("status") == "skipped":
        print(f"  └─ Reason: {heber_freshness.get('reason', 'unknown')}")
    else:
        stale_feeds = heber_freshness.get("stale_feeds", [])
        if stale_feeds:
            print(f"  └─ Stale Feeds: {', '.join(stale_feeds)}")
        elif heber_freshness.get("reason"):
            print(f"  └─ Reason: {heber_freshness.get('reason')}")
    print()

    # Overall
    overall = results.get("overall_status", "unknown")
    overall_symbol = "✓" if overall == "ok" else "✗"
    print("=" * 60)
    print(f"{overall_symbol} Overall Status: {overall.upper()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_healthcheck(verbose=True)
