#!/usr/bin/env python3
"""
Migration: Add Historical Replay Tables

This script creates the new tables required for the Historical Replay Data Architecture:
- external_snapshots: Raw external API data (GEX, flow)
- feature_snapshots: Computed features at point-in-time
- daily_universe: Track filtered symbols per day

Run this script once to add the tables to an existing database.

Usage:
    python scripts/migrate_add_replay_tables.py

Or simply call init_db() which uses create_all() and is idempotent.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Base
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


def migrate():
    """Create the new replay tables."""
    logger = StructuredLogger("migrate_replay_tables")
    config_loader = ConfigLoader()

    # Load config
    config = config_loader.load_config()

    # Initialize database
    db = DatabaseDatabase(config_loader, logger, config)

    logger.info("Starting migration: Add Historical Replay Tables")

    # Check which tables already exist
    with db.engine.connect() as conn:
        inspector_result = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
        existing_tables = {row[0] for row in inspector_result}

    tables_to_create = []

    if "external_snapshots" not in existing_tables:
        tables_to_create.append("external_snapshots")
    else:
        logger.info("Table 'external_snapshots' already exists, skipping")

    if "feature_snapshots" not in existing_tables:
        tables_to_create.append("feature_snapshots")
    else:
        logger.info("Table 'feature_snapshots' already exists, skipping")

    if "daily_universe" not in existing_tables:
        tables_to_create.append("daily_universe")
    else:
        logger.info("Table 'daily_universe' already exists, skipping")

    if not tables_to_create:
        logger.info("All replay tables already exist. Migration complete.")
        return

    # Create missing tables
    logger.info(f"Creating tables: {tables_to_create}")

    # Use create_all which is idempotent (only creates tables that don't exist)
    Base.metadata.create_all(bind=db.engine)

    logger.info("Migration complete: Historical Replay Tables created")

    # Verify
    with db.engine.connect() as conn:
        for table in tables_to_create:
            result = conn.execute(
                text(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
                )
            )
            if result.fetchone():
                logger.info(f"✓ Table '{table}' created successfully")
            else:
                logger.error(f"✗ Failed to create table '{table}'")


if __name__ == "__main__":
    migrate()
