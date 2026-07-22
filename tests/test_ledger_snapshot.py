"""Unit tests for the ledger snapshot exporter + corruption guard."""

import sqlite3

import pytest

import src.ops.ledger_snapshot as ls


def _make_db(path, rows=3):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(f"row{i}",) for i in range(rows)])
    conn.commit()
    conn.close()


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    state = tmp_path / "state"
    export = tmp_path / "export"
    state.mkdir()
    monkeypatch.setattr(ls, "STATE_DIR", state)
    monkeypatch.setattr(ls, "EXPORT_DIR", export)
    return state, export


@pytest.mark.unit
def test_snapshot_exports_consistent_copy(dirs):
    state, export = dirs
    _make_db(state / "ledger.db", rows=5)

    assert ls.snapshot_db("ledger.db") is True

    dst = export / "ledger.db"
    assert dst.exists()
    conn = sqlite3.connect(dst)
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 5
    conn.close()


@pytest.mark.unit
def test_corrupt_source_preserves_last_good_snapshot(dirs):
    state, export = dirs
    # First a healthy snapshot exists.
    _make_db(state / "ledger.db", rows=2)
    assert ls.snapshot_db("ledger.db") is True
    good = (export / "ledger.db").read_bytes()

    # Now the live DB is corrupt ("file is not a database").
    (state / "ledger.db").write_bytes(b"not a sqlite database at all")
    assert ls.snapshot_db("ledger.db") is False
    # Last good export is untouched — readers keep working, alert fires in logs.
    assert (export / "ledger.db").read_bytes() == good


@pytest.mark.unit
def test_oversized_db_is_not_exported(dirs, monkeypatch):
    state, export = dirs
    _make_db(state / "ledger.db")
    monkeypatch.setattr(ls, "MAX_DB_BYTES", 1)  # anything is "too big"

    assert ls.integrity_ok(state / "ledger.db") is False
    assert ls.snapshot_db("ledger.db") is False
    assert not (export / "ledger.db").exists()


@pytest.mark.unit
def test_missing_source_is_not_an_export(dirs):
    _state, export = dirs
    assert ls.snapshot_db("ledger.db") is False
    assert not (export / "ledger.db").exists()
