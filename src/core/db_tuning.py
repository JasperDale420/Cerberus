"""SQLite connection tuning shared by the ledger and analytics engines."""

from __future__ import annotations

import sqlite3

from sqlalchemy import event
from sqlalchemy.engine import Engine


def apply_sqlite_pragmas(engine: Engine, *, wal: bool = True, busy_timeout_ms: int = 5000) -> None:
    """Set multi-process-safe pragmas on every new connection of ``engine``.

    ``busy_timeout`` is the safety-critical one and is always applied: without it a
    writer blocked by a concurrent reader (the snapshot exporter reading the live DB)
    fails immediately with "database is locked" instead of waiting, which can drop a
    trade record.

    ``journal_mode=WAL`` is best-effort here: SQLite only switches when it can get the
    database exclusively, so under connection pooling it may silently stay in the
    current mode (WAL persists in the header once set — the migration sets it with
    exclusive access). WAL lets the single writer and readers proceed concurrently.
    Only safe on a native filesystem (Docker named volume) — WAL over a macOS bind
    mount is the corruption case this whole change exists to avoid.
    """
    if engine.url.get_backend_name() != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _record):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        try:
            cur.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            cur.execute("PRAGMA synchronous=NORMAL")
            if wal:
                try:
                    cur.execute("PRAGMA journal_mode=WAL")
                except sqlite3.OperationalError:
                    # Contended switch — never let it break connection creation.
                    pass
        finally:
            cur.close()
