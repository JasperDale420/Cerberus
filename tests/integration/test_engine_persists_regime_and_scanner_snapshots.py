from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import RegimeHistory, ScannerSnapshot
from src.core.config import ConfigLoader
from src.core.domain import (
    Bar,
    Regime,
    ScanResult,
    SymbolFeatures,
    SymbolState,
    WatchlistSymbol,
)
from src.core.logger import StructuredLogger
from src.engine.execution import ExecutionEngine


@pytest.mark.integration
def test_engine_persists_regime_history_on_index_bar(tmp_path: Path) -> None:
    db_path = tmp_path / "engine.db"
    db_url = f"sqlite:///{db_path}"

    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url, "index_symbol": "SPY"}
    logger = StructuredLogger("TestRegimeHistory", level="INFO")
    db = DatabaseDatabase(loader, logger)
    db.init_db()

    engine = ExecutionEngine(config={"index_symbol": "SPY"}, logger=logger, db=db)
    engine.on_bar(
        "SPY",
        Bar(
            symbol="SPY",
            time=datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000.0,
        ),
    )

    with db.get_session() as session:
        rows = session.query(RegimeHistory).all()
        assert len(rows) == 1
        assert rows[0].index_symbol == "SPY"
        assert rows[0].regime in {"bull", "bear", "chop"}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_engine_persists_scanner_snapshots_on_run_scan(tmp_path: Path) -> None:
    db_path = tmp_path / "engine.db"
    db_url = f"sqlite:///{db_path}"

    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url, "index_symbol": "SPY"}
    logger = StructuredLogger("TestScannerSnapshots", level="INFO")
    db = DatabaseDatabase(loader, logger)
    db.init_db()

    engine = ExecutionEngine(config={"index_symbol": "SPY"}, logger=logger, db=db)

    # Minimal state for apply_scan_result add path.
    engine.symbol_states["SPY"] = SymbolState(
        symbol="SPY",
        bars=deque(maxlen=100),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    now = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
    features = SymbolFeatures(
        symbol="AAPL",
        price=100.0,
        atr_pct=0.01,
        avg_volume=1_000_000.0,
        intraday_range_pct=0.01,
        gap_pct=0.02,
        ema20_slope=0.0,
        ema_trend_strength=0.0,
        distance_from_vwap=0.0,
        premarket_volume=0.0,
        adx=25.0,
        distance_from_ema20=0.0,
        prior_day_high=101.0,
        prior_day_low=99.0,
        bb_upper=102.0,
        bb_lower=98.0,
        price_zscore=1.0,
        flow_zscore=0.0,
        call_put_ratio=0.0,
        large_sweeps_count=0,
        aggressive_flow_share=0.0,
        last_updated=now,
        extra={"nested_time": now},
    )
    scan_result = ScanResult(
        generated_at=now,
        regime=Regime.CHOP,
        watchlist=[
            WatchlistSymbol(
                symbol="AAPL",
                score=1.0,
                strategies=["vwap_reversion"],
                features=features,
            )
        ],
    )

    engine.scanner = MagicMock()
    engine.scanner.scan = AsyncMock(return_value=scan_result)

    await engine.run_scan()

    assert isinstance(engine.symbol_states["AAPL"].meta.get("features_snapshot"), dict)
    sym_snap = engine.symbol_states["AAPL"].meta["features_snapshot"]
    assert isinstance(sym_snap.get("last_updated"), str)
    assert isinstance(sym_snap.get("extra"), dict)
    assert isinstance(sym_snap["extra"].get("nested_time"), str)

    with db.get_session() as session:
        rows = session.query(ScannerSnapshot).all()
        assert len(rows) == 1
        assert rows[0].symbol == "AAPL"
        assert rows[0].scanner_score == pytest.approx(1.0)
        assert isinstance(rows[0].features_json, dict)
        assert isinstance(rows[0].features_json.get("last_updated"), str)
        assert isinstance(rows[0].features_json.get("extra"), dict)
        assert isinstance(rows[0].features_json["extra"].get("nested_time"), str)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_engine_run_scan_degrades_gracefully_on_scan_error(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "engine.db"
    db_url = f"sqlite:///{db_path}"

    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url, "index_symbol": "SPY"}
    logger = StructuredLogger("TestScanError", level="INFO")
    db = DatabaseDatabase(loader, logger)
    db.init_db()

    engine = ExecutionEngine(config={"index_symbol": "SPY"}, logger=logger, db=db)

    # Minimal state required by apply_scan_result (index symbol must exist).
    engine.symbol_states["SPY"] = SymbolState(
        symbol="SPY",
        bars=deque(maxlen=100),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    engine.scanner = MagicMock()
    engine.scanner.scan = AsyncMock(side_effect=RuntimeError("boom"))

    await engine.run_scan()
