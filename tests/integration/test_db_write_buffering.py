from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Signal as DbSignal
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


@pytest.mark.integration
def test_db_write_buffers_failed_writes_and_flushes_later(tmp_path: Path) -> None:
    db_path = tmp_path / "buffer.db"
    db_url = f"sqlite:///{db_path}"

    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {
        "database_url": db_url,
        "db_write_buffer_max": 10,
        "db_write_flush_max": 10,
        "db_fail_mode": "warn",
    }
    logger = StructuredLogger("TestDBBuffer", level="INFO")
    db = DatabaseDatabase(loader, logger)
    db.init_db()

    original_get_session = db.get_session

    @contextmanager
    def failing_session():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    db.get_session = failing_session  # type: ignore[assignment]

    ok = db.write(
        "signal",
        lambda session: session.add(
            DbSignal(
                correlation_id="c1",
                symbol="SPY",
                strategy="s",
                regime="chop",
                time=datetime.now(timezone.utc),
                raw_side="buy",
                raw_size=1.0,
                accepted=True,
                rejection_reason=None,
                meta_json=None,
            )
        ),
    )
    assert ok is False
    assert db.last_db_write_error is not None
    assert len(db._write_buffer) == 1  # type: ignore[attr-defined]

    db.get_session = original_get_session  # type: ignore[assignment]
    flushed = db.flush_writes()
    assert flushed == 1

    with db.get_session() as session:
        rows = session.query(DbSignal).all()
        assert len(rows) == 1
        assert rows[0].correlation_id == "c1"


@pytest.mark.integration
def test_db_write_buffer_drops_when_full(tmp_path: Path) -> None:
    db_path = tmp_path / "buffer_full.db"
    db_url = f"sqlite:///{db_path}"

    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {
        "database_url": db_url,
        "db_write_buffer_max": 2,
        "db_write_flush_max": 10,
        "db_fail_mode": "warn",
    }
    logger = StructuredLogger("TestDBBufferFull", level="INFO")
    db = DatabaseDatabase(loader, logger)

    @contextmanager
    def failing_session():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    db.get_session = failing_session  # type: ignore[assignment]

    assert db.write("noop", lambda session: None) is False
    assert db.write("noop", lambda session: None) is False
    assert len(db._write_buffer) == 2  # type: ignore[attr-defined]

    # Third write should be dropped (buffer already full).
    assert db.write("noop", lambda session: None) is False
    assert len(db._write_buffer) == 2  # type: ignore[attr-defined]


@pytest.mark.integration
def test_db_flush_stops_on_first_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "flush_stop.db"
    db_url = f"sqlite:///{db_path}"

    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {
        "database_url": db_url,
        "db_write_buffer_max": 10,
        "db_write_flush_max": 10,
        "db_fail_mode": "warn",
    }
    logger = StructuredLogger("TestDBFlushStop", level="INFO")
    db = DatabaseDatabase(loader, logger)

    db._write_buffer.append(("ok", lambda session: None))  # type: ignore[attr-defined]

    def boom(session):
        raise RuntimeError("boom")

    db._write_buffer.append(("boom", boom))  # type: ignore[attr-defined]
    db._write_buffer.append(("never", lambda session: None))  # type: ignore[attr-defined]

    flushed = db.flush_writes()
    assert flushed == 1
    assert len(db._write_buffer) == 2  # type: ignore[attr-defined]

    flushed_again = db.flush_writes()
    assert flushed_again == 0
    assert len(db._write_buffer) == 2  # type: ignore[attr-defined]
