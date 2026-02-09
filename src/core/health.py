"""
Healthcheck module for Cerberus trading system.

Provides system health verification for monitoring and operational readiness.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

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
    from src.core.settings import get_settings

    settings = get_settings()
    api_key = settings.resolved_api_key
    secret_key = settings.resolved_secret_key
    base_url = settings.resolved_base_url

    if not api_key or not secret_key:
        return {
            "status": "error",
            "error": "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY environment variables",
        }

    is_paper = settings.alpaca_paper
    mode = "paper" if is_paper else "live"

    return {
        "status": "ok",
        "mode": mode,
        "base_url": base_url or "not_set",
        "credentials_present": True,
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
    }

    # Determine overall health
    all_ok = all(
        component.get("status") == "ok"
        for key, component in results.items()
        if key != "system" and isinstance(component, dict)
    )
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
    print(
        f"{status_symbol} Database connectivity: {db.get('status', 'unknown').upper()}"
    )
    if db.get("status") != "ok":
        print(f"  └─ Error: {db.get('error', 'unknown')}")
    else:
        print(f"  └─ Path: {db.get('path', 'unknown')}")
    print()

    # Alpaca
    alpaca = results.get("alpaca", {})
    status_symbol = "✓" if alpaca.get("status") == "ok" else "✗"
    print(f"{status_symbol} Alpaca API: {alpaca.get('status', 'unknown').upper()}")
    if alpaca.get("status") != "ok":
        print(f"  └─ Error: {alpaca.get('error', 'unknown')}")
    else:
        print(f"  └─ Mode: {alpaca.get('mode', 'unknown')}")
        print(f"  └─ Base URL: {alpaca.get('base_url', 'unknown')}")
    print()

    # Overall
    overall = results.get("overall_status", "unknown")
    overall_symbol = "✓" if overall == "ok" else "✗"
    print("=" * 60)
    print(f"{overall_symbol} Overall Status: {overall.upper()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_healthcheck(verbose=True)
