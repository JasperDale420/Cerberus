"""Snapshot persistence helpers for external data and computed feature state."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.domain import SymbolFeatures


@dataclass(slots=True)
class ExternalSnapshotRecord:
    source: str
    symbol: str
    snapshot_time: datetime
    data: Any
    git_sha: str | None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class FeatureSnapshotRecord:
    symbol: str
    as_of_ts: datetime
    features: dict[str, Any]
    git_sha: str | None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class DailyUniverseRecord:
    trade_date: datetime
    symbol: str
    source: str
    git_sha: str | None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _as_utc(ts: datetime) -> datetime:
    """Normalize timestamp to timezone-aware UTC for storage consistency."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _get_git_sha() -> str | None:
    """Return short git SHA when available; None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        return sha or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class SnapshotManager:
    """Persist serialized snapshots via the repository's DB session API."""

    def __init__(self, db: Any, logger: Any):
        self.db = db
        self.logger = logger
        self._git_sha = _get_git_sha()

    def persist_external_snapshot(
        self,
        *,
        source: str,
        symbol: str,
        snapshot_time: datetime,
        data: Any,
    ) -> None:
        record = ExternalSnapshotRecord(
            source=source,
            symbol=symbol,
            snapshot_time=_as_utc(snapshot_time),
            data=data,
            git_sha=self._git_sha,
        )
        try:
            with self.db.get_session() as session:
                session.add(record)
                session.commit()
            self.logger.info(f"Persisted external snapshot for {source}:{symbol}")
        except Exception:
            self.logger.error(
                f"Failed to persist external snapshot for {source}:{symbol}",
                exc_info=True,
            )
            raise

    def persist_feature_snapshot(self, *, features: SymbolFeatures, as_of_ts: datetime) -> None:
        record = FeatureSnapshotRecord(
            symbol=features.symbol,
            as_of_ts=_as_utc(as_of_ts),
            features=asdict(features),
            git_sha=self._git_sha,
        )
        try:
            with self.db.get_session() as session:
                session.add(record)
                session.commit()
            self.logger.info(f"Persisted feature snapshot for {features.symbol}")
        except Exception:
            self.logger.error(
                f"Failed to persist feature snapshot for {features.symbol}",
                exc_info=True,
            )
            raise

    def persist_daily_universe(self, *, trade_date: datetime, symbols: list[str], source: str) -> None:
        if not symbols:
            self.logger.warning("Skipped daily universe persistence; no symbols", source=source)
            return

        trade_ts = _as_utc(trade_date)
        try:
            with self.db.get_session() as session:
                for symbol in symbols:
                    session.add(
                        DailyUniverseRecord(
                            trade_date=trade_ts,
                            symbol=symbol,
                            source=source,
                            git_sha=self._git_sha,
                        )
                    )
                session.commit()
            self.logger.info("Persisted daily universe", source=source, symbol_count=len(symbols))
        except Exception:
            self.logger.error(
                f"Failed to persist daily universe for {source}",
                exc_info=True,
            )
            raise
