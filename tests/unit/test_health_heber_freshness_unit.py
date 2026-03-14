from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core import health


def _touch(path: Path, ts: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")
    epoch = ts.timestamp()
    os.utime(path, (epoch, epoch))


@pytest.mark.unit
def test_check_heber_freshness_reports_recent_dataset_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    bars_file = tmp_path / "silver" / "feed=bars" / "instrument_type=equity" / "dt=2026-02-10" / "part-bars.parquet"
    trades_file = (
        tmp_path
        / "silver"
        / "feed=trades"
        / "instrument_type=equity"
        / "dt=2026-02-10"
        / "hour=14"
        / "part-trades.parquet"
    )
    _touch(bars_file, now)
    _touch(trades_file, now)

    settings = SimpleNamespace(
        cerberus_heber_data_root=str(tmp_path),
    )
    monkeypatch.setattr(health, "get_settings", lambda: settings)

    result = health.check_heber_freshness(required_feeds=("bars", "trades"), max_age_seconds=300)

    assert result["status"] == "ok"
    assert result["stale_feeds"] == []
    assert set(result["freshness"].keys()) == {"bars", "trades"}


@pytest.mark.unit
def test_check_heber_freshness_reports_stale_when_no_recent_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    bars_file = tmp_path / "silver" / "feed=bars" / "instrument_type=equity" / "dt=2026-01-01" / "part-bars.parquet"
    _touch(bars_file, old_time)

    settings = SimpleNamespace(
        cerberus_heber_data_root=str(tmp_path),
    )
    monkeypatch.setattr(health, "get_settings", lambda: settings)

    result = health.check_heber_freshness(required_feeds=("bars", "trades"), max_age_seconds=60)

    assert result["status"] == "degraded"
    assert "bars" in result["stale_feeds"]
    assert "trades" in result["stale_feeds"]
