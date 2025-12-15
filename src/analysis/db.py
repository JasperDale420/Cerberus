import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.analysis.schema import Base
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


class DatabaseDatabase:
    def __init__(self, config_loader: ConfigLoader, logger: StructuredLogger):
        self.config = config_loader.load_config()
        self.logger = logger

        # Default to local SQLite if not configured
        db_url = self.config.get("database_url", "sqlite:///cerberus.db")

        # Ensure sqlite path is absolute if relative
        if db_url.startswith("sqlite:///"):
            path = db_url.replace("sqlite:///", "")
            if not os.path.isabs(path) and path != ":memory:":
                # Make it relative to project root or configured data dir
                # For now, just leave it as CWD relative or handle it
                pass

        self.engine = create_engine(
            db_url, echo=self.config.get("db_echo", False), pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self.logger.info("Database engine initialized", url=db_url)

    def init_db(self):
        """
        Creates all tables.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables verified/created")
        except Exception as e:
            self.logger.error("Failed to initialize database", error=str(e))
            raise

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
