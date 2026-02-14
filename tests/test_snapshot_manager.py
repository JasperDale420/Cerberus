"""
Unit tests for snapshot capture and replay functionality.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.core.domain import SymbolFeatures
from src.data.snapshot_manager import SnapshotManager


@pytest.fixture
def mock_db():
    """Create a mock database with session context manager."""
    db = MagicMock()
    session = MagicMock()
    db.get_session.return_value.__enter__ = MagicMock(return_value=session)
    db.get_session.return_value.__exit__ = MagicMock(return_value=False)
    return db


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def snapshot_manager(mock_db, mock_logger):
    return SnapshotManager(mock_db, mock_logger)


def test_persist_external_snapshot_gex(snapshot_manager, mock_db):
    """Test persisting GEX data."""
    gex_data = [
        {"strike": 500, "call_gamma": 100, "put_gamma": 50},
        {"strike": 510, "call_gamma": 120, "put_gamma": 60},
    ]

    snapshot_manager.persist_external_snapshot(
        source="gex",
        symbol="SPY",
        snapshot_time=datetime(2026, 1, 13, 14, 35, tzinfo=timezone.utc),
        data=gex_data,
    )

    # Verify session.add was called
    session = mock_db.get_session.return_value.__enter__.return_value
    assert session.add.called
    assert session.commit.called


def test_persist_feature_snapshot(snapshot_manager, mock_db):
    """Test persisting computed features."""
    features = SymbolFeatures(
        symbol="AAPL",
        price=185.50,
        atr_pct=0.015,
        avg_volume=50000000.0,
        intraday_range_pct=0.02,
        gap_pct=0.01,
        ema20_slope=0.5,
        ema_trend_strength=1.2,
        distance_from_vwap=0.005,
        premarket_volume=1000000.0,
        adx=25.0,
        distance_from_ema20=0.008,
        prior_day_high=186.0,
        prior_day_low=184.0,
        bb_upper=187.0,
        bb_lower=183.0,
        price_zscore=0.5,
        flow_zscore=1.5,
        call_put_ratio=1.2,
        large_sweeps_count=5,
        aggressive_flow_share=0.6,
        last_updated=datetime(2026, 1, 13, 14, 35, tzinfo=timezone.utc),
        tfi=0.75,
        hurst_exponent=0.65,
        frac_diff_close=0.02,
        net_gex=50000.0,
        gex_flip_dist=0.03,
    )

    snapshot_manager.persist_feature_snapshot(
        features=features,
        as_of_ts=datetime(2026, 1, 13, 14, 35, tzinfo=timezone.utc),
    )

    session = mock_db.get_session.return_value.__enter__.return_value
    assert session.add.called
    assert session.commit.called


def test_persist_daily_universe(snapshot_manager, mock_db):
    """Test persisting daily filtered universe."""
    symbols = ["AAPL", "NVDA", "TSLA"]

    snapshot_manager.persist_daily_universe(
        trade_date=datetime(2026, 1, 13, tzinfo=timezone.utc),
        symbols=symbols,
        source="scanner",
    )

    session = mock_db.get_session.return_value.__enter__.return_value
    # Should add one entry per symbol
    assert session.add.call_count == 3
    assert session.commit.called


def test_persist_daily_universe_skips_empty(snapshot_manager, mock_db, mock_logger):
    """Empty universes should skip persistence to avoid empty writes."""
    snapshot_manager.persist_daily_universe(
        trade_date=datetime(2026, 1, 13, tzinfo=timezone.utc),
        symbols=[],
        source="scanner",
    )

    session = mock_db.get_session.return_value.__enter__.return_value
    assert not session.add.called
    assert not session.commit.called
    mock_logger.warning.assert_called_once()


def test_git_sha_retrieval():
    """Test git SHA retrieval for code versioning."""
    from src.data.snapshot_manager import _get_git_sha

    sha = _get_git_sha()
    # Should either return a short SHA or None (if not in git repo)
    assert sha is None or (isinstance(sha, str) and len(sha) <= 12)
