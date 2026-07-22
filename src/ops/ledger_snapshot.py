"""Consistent SQLite snapshot exporter + corruption guard.

The live ``ledger.db`` / ``cerberus.db`` are written by the trading containers on a
Docker named volume (native filesystem, reliable POSIX locking). Host tools (the
EmpireUI dashboard, Athena, Heber, ``scripts/ledger_audit.py``) can't read a named
volume directly on macOS, so this exporter publishes a consistent copy to a
host-visible directory via ``VACUUM INTO``.

Every cycle also runs ``PRAGMA integrity_check`` and a size sanity check on the live
DB. If the DB is corrupt or has ballooned, it logs an error and refuses to overwrite
the last good snapshot — turning the silent multi-month corruption that motivated this
module into a loud log line within one interval.

Run as a sidecar loop::

    python -m src.ops.ledger_snapshot --loop --interval 900

Or once (health check / CI)::

    python -m src.ops.ledger_snapshot --once
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from pathlib import Path

from src.core.logger import get_logger

logger = get_logger("cerberus.ledger_snapshot")

# Live DBs (in the named volume) mapped to their host-visible snapshot path.
# Overridable via env for local runs / testing.
STATE_DIR = Path(os.getenv("CERBERUS_STATE_DIR", "/app/state"))
EXPORT_DIR = Path(os.getenv("CERBERUS_EXPORT_DIR", "/app/export"))
DB_NAMES = tuple(os.getenv("CERBERUS_SNAPSHOT_DBS", "ledger.db,cerberus.db").split(","))

# Above this the DB is treated as corrupt/pathological and NOT exported. The incident
# that motivated this module ballooned a trade ledger to 18.5 GB.
MAX_DB_BYTES = int(os.getenv("CERBERUS_SNAPSHOT_MAX_BYTES", str(2 * 1024**3)))


def integrity_ok(db_path: Path) -> bool:
    """Return True iff PRAGMA integrity_check passes and size is sane."""
    size = db_path.stat().st_size
    if size > MAX_DB_BYTES:
        logger.error(
            "ledger_snapshot_size_anomaly",
            db=str(db_path),
            size_bytes=size,
            max_bytes=MAX_DB_BYTES,
        )
        return False
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            result = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        logger.error("ledger_snapshot_integrity_error", db=str(db_path), error=str(exc), exc_info=True)
        return False
    if not result or result[0] != "ok":
        logger.error("ledger_snapshot_integrity_failed", db=str(db_path), detail=str(result))
        return False
    return True


def snapshot_db(name: str) -> bool:
    """Integrity-check the live DB and export a consistent copy. Returns success."""
    src = STATE_DIR / name
    dst = EXPORT_DIR / name
    if not src.exists():
        logger.warning("ledger_snapshot_source_missing", db=str(src))
        return False
    if not integrity_ok(src):
        # Leave the last good snapshot in place so readers keep working.
        return False

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EXPORT_DIR / f".{name}.tmp"
    if tmp.exists():
        tmp.unlink()
    try:
        conn = sqlite3.connect(src, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("VACUUM INTO ?", (str(tmp),))
        finally:
            conn.close()
        os.replace(tmp, dst)  # atomic within the export dir
    except (sqlite3.DatabaseError, OSError) as exc:
        logger.error("ledger_snapshot_export_failed", db=name, error=str(exc), exc_info=True)
        if tmp.exists():
            tmp.unlink()
        return False
    logger.info("ledger_snapshot_exported", db=name, dst=str(dst), size_bytes=dst.stat().st_size)
    return True


def run_once() -> bool:
    """Snapshot every configured DB. Returns True iff all succeeded."""
    return all([snapshot_db(name.strip()) for name in DB_NAMES if name.strip()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export consistent SQLite snapshots + integrity guard")
    parser.add_argument("--loop", action="store_true", help="Run forever, sleeping --interval between cycles")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--interval", type=int, default=int(os.getenv("CERBERUS_SNAPSHOT_INTERVAL", "900")))
    args = parser.parse_args()

    if not args.loop:
        ok = run_once()
        raise SystemExit(0 if ok else 1)

    logger.info("ledger_snapshot_loop_started", interval_s=args.interval, dbs=list(DB_NAMES))
    while True:
        try:
            run_once()
        except Exception:  # keep the loop alive; a bad cycle must not kill the guard
            logger.error("ledger_snapshot_cycle_error", exc_info=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
