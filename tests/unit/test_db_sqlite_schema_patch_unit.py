from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.analysis.db import DatabaseDatabase
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


def _columns_for_table(db_path: Path, table_name: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


@pytest.mark.unit
def test_init_db_patches_known_sqlite_schema_gaps(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_cerberus.db"
    db_url = f"sqlite:///{db_path}"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                strategy TEXT,
                regime_at_entry TEXT,
                regime_at_exit TEXT,
                side TEXT,
                qty REAL,
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                commission REAL,
                slippage_estimate REAL,
                pnl_gross REAL,
                pnl_net REAL,
                initial_risk REAL,
                pnl_r REAL,
                mae_r REAL,
                mfe_r REAL,
                holding_period_seconds REAL,
                features_json TEXT,
                correlation_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE signals (
                id INTEGER PRIMARY KEY,
                correlation_id TEXT,
                symbol TEXT,
                strategy TEXT,
                regime TEXT,
                time TEXT,
                raw_side TEXT,
                raw_size REAL,
                accepted INTEGER,
                rejection_reason TEXT,
                meta_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE regime_history (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                regime TEXT,
                index_symbol TEXT,
                index_price REAL,
                cum_ret REAL,
                trend_score REAL,
                vol REAL
            )
            """
        )
        conn.commit()

    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url}
    logger = StructuredLogger("TestDBSchemaPatch", level="INFO")

    db = DatabaseDatabase(loader, logger)
    db.init_db()

    trade_columns = _columns_for_table(db_path, "trades")
    assert "regime_tags_entry_json" in trade_columns
    assert "regime_tags_exit_json" in trade_columns

    signal_columns = _columns_for_table(db_path, "signals")
    assert "feature_snapshot_json" in signal_columns

    regime_columns = _columns_for_table(db_path, "regime_history")
    assert "model_version" in regime_columns
    assert "trend" in regime_columns
    assert "vol_regime" in regime_columns
    assert "liquidity" in regime_columns
    assert "risk" in regime_columns
    assert "session" in regime_columns
    assert "vol_of_vol" in regime_columns
    assert "liquidity_score" in regime_columns
    assert "risk_score" in regime_columns
    assert "confidence_json" in regime_columns
