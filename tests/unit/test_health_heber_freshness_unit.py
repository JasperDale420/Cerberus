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


@pytest.mark.unit
def test_latest_dataset_file_does_not_stat_every_dt_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for healthcheck timeout: with 2,500+ dt= partitions on a
    bind-mounted volume each is_dir() costs ~30 ms, so iterating them all
    blew past the 60 s healthcheck timeout. Verify is_dir() is called only
    for the small set of top-N candidates we actually look at, not for
    every sibling.
    """
    feed_root = tmp_path / "silver" / "feed=bars" / "instrument_type=equity"
    feed_root.mkdir(parents=True)
    # 100 stand-in date partitions plus one real one with content.
    for i in range(100):
        (feed_root / f"dt=2024-01-{i:02d}").mkdir()
    real_partition = feed_root / "dt=2026-04-28"
    real_partition.mkdir()
    real_file = real_partition / "part-bars.parquet"
    _touch(real_file, datetime.now(UTC))

    is_dir_calls: list[str] = []
    original_is_dir = Path.is_dir

    def counting_is_dir(self: Path, *args, **kwargs):  # noqa: ANN002,ANN003
        is_dir_calls.append(self.name)
        return original_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(health.Path, "is_dir", counting_is_dir)

    result = health._latest_dataset_file(tmp_path, "bars")

    assert result == real_file
    # We have ~101 dt= dirs; the optimization should call is_dir on at
    # most the top-3 candidates (the rglob below also calls is_dir on
    # the discovered file, which is a separate concern). Anything
    # remotely near 100 means we regressed.
    assert len(is_dir_calls) < 10, f"is_dir called {len(is_dir_calls)} times — regression!"
