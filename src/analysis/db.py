import os
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable, Deque, Dict, Generator, Optional, Tuple

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from src.analysis.schema import Base
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


class DatabaseDatabase:
    SQLITE_SCHEMA_PATCHES: Dict[str, tuple[tuple[str, str], ...]] = {
        "trades": (
            ("regime_tags_entry_json", "TEXT"),
            ("regime_tags_exit_json", "TEXT"),
        ),
        "signals": (("feature_snapshot_json", "TEXT"),),
        "regime_history": (
            ("model_version", "TEXT"),
            ("trend", "TEXT"),
            ("vol_regime", "TEXT"),
            ("liquidity", "TEXT"),
            ("risk", "TEXT"),
            ("session", "TEXT"),
            ("vol_of_vol", "REAL"),
            ("liquidity_score", "REAL"),
            ("risk_score", "REAL"),
            ("confidence_json", "TEXT"),
        ),
    }

    def __init__(
        self,
        config_loader: ConfigLoader,
        logger: StructuredLogger,
        config: Optional[Dict[str, Any]] = None,
        config_path_or_dir: Optional[str] = None,
    ):
        self.config = config if isinstance(config, dict) else config_loader.load_config(config_path_or_dir)
        self.logger = logger

        # PRD 11.4: bounded in-memory buffering for transient DB failures.
        self.db_write_buffer_max: int = int(self.config.get("db_write_buffer_max", 200))
        self.db_write_flush_max: int = int(self.config.get("db_write_flush_max", 50))
        self.db_fail_mode: str = str(self.config.get("db_fail_mode", "warn")).lower()
        self._write_buffer: Deque[Tuple[str, Callable[[Session], None]]] = deque()
        self.last_db_write_error: Optional[str] = None

        # Default to local SQLite if not configured
        db_url = self.config.get("database_url", "sqlite:///cerberus.db")

        # Ensure sqlite path is absolute if relative
        if db_url.startswith("sqlite:///"):
            path = db_url.replace("sqlite:///", "")
            if not os.path.isabs(path) and path != ":memory:":
                # Make it relative to project root or configured data dir
                # For now, just leave it as CWD relative or handle it
                pass

        self.engine = create_engine(db_url, echo=self.config.get("db_echo", False), pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.logger.info("Database engine initialized", url=db_url)

    def init_db(self):
        """
        Creates all tables.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            self._apply_sqlite_schema_patches()
            self.logger.info("Database tables verified/created")
        except Exception as e:
            self.logger.error("Failed to initialize database", error=str(e))
            raise

    def _apply_sqlite_schema_patches(self) -> None:
        """Patch known legacy SQLite schemas that predate newer ORM columns."""
        if self.engine.url.get_backend_name() != "sqlite":
            return

        with self.engine.begin() as conn:
            inspector = inspect(conn)
            for table_name, columns in self.SQLITE_SCHEMA_PATCHES.items():
                if not inspector.has_table(table_name):
                    continue
                existing = {str(col.get("name")) for col in inspector.get_columns(table_name) if col.get("name")}
                for column_name, column_type in columns:
                    if column_name in existing:
                        continue
                    conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {column_type}'))
                    self.logger.warning(
                        "Applied sqlite schema patch",
                        table=table_name,
                        column=column_name,
                        column_type=column_type,
                    )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error("Database session rollback", error=str(e))
            raise
        finally:
            session.close()

    def write(self, kind: str, fn: Callable[[Session], None]) -> bool:
        """
        Best-effort write helper with bounded buffering.

        - On success: executes and opportunistically flushes buffered writes.
        - On failure: buffers the write closure (bounded) unless db_fail_mode == 'raise'.
        """
        try:
            with self.get_session() as session:
                fn(session)

            # Opportunistically flush any buffered writes after a successful write.
            if self._write_buffer:
                self.flush_writes()

            self.last_db_write_error = None
            return True
        except Exception as e:
            self.last_db_write_error = str(e)

            if self.db_fail_mode == "raise":
                raise

            if len(self._write_buffer) >= self.db_write_buffer_max:
                self.logger.error(
                    "DB write buffer full; dropping write",
                    kind=kind,
                    buffer_len=len(self._write_buffer),
                    buffer_max=self.db_write_buffer_max,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return False

            self._write_buffer.append((kind, fn))
            log_fn = self.logger.warning if self.db_fail_mode == "warn" else self.logger.error
            log_fn(
                "DB write failed; buffered for retry",
                kind=kind,
                buffer_len=len(self._write_buffer),
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    def flush_writes(self) -> int:
        """
        Attempts to flush buffered writes in FIFO order.
        Stops early on first failure to avoid busy-looping on persistent outages.
        """
        flushed = 0
        to_flush = min(len(self._write_buffer), self.db_write_flush_max)
        for _ in range(to_flush):
            kind, fn = self._write_buffer[0]
            try:
                with self.get_session() as session:
                    fn(session)
                self._write_buffer.popleft()
                flushed += 1
            except Exception as e:
                self.last_db_write_error = str(e)
                self.logger.error(
                    "DB flush failed; will retry later",
                    kind=kind,
                    buffer_len=len(self._write_buffer),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                break

        return flushed

    def write_buffer_len(self) -> int:
        return len(self._write_buffer)

    def write_buffer_max(self) -> int:
        return int(self.db_write_buffer_max)
