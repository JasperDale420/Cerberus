from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Signal as DbSignal
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


@pytest.mark.integration
def test_database_session_commits_on_success(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"

    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url}
    logger = StructuredLogger("TestDBCommit", level="INFO")

    db = DatabaseDatabase(loader, logger)
    db.init_db()

    with db.get_session() as session:
        session.add(
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
        )

    with db.get_session() as session:
        rows = session.query(DbSignal).all()
        assert len(rows) == 1
        assert rows[0].correlation_id == "c1"


@pytest.mark.integration
def test_database_session_rolls_back_on_error(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"

    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url}
    logger = StructuredLogger("TestDBRollback", level="INFO")

    db = DatabaseDatabase(loader, logger)
    db.init_db()

    with pytest.raises(RuntimeError):
        with db.get_session() as session:
            session.add(
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
            )
            raise RuntimeError("boom")

    with db.get_session() as session:
        rows = session.query(DbSignal).all()
        assert rows == []
