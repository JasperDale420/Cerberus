from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.analysis.schema import Trade as DbTrade
from src.core.config import ConfigLoader
from src.core.domain import Bar, MarketState, Position, Regime, Side, SymbolState
from src.core.logger import StructuredLogger
from src.engine.execution import ExecutionEngine


def _make_db(tmp_path: Path) -> DatabaseDatabase:
    db_path = tmp_path / "t.db"
    db_url = f"sqlite:///{db_path}"
    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url}
    logger = StructuredLogger("TestDB", level="INFO")
    db = DatabaseDatabase(loader, logger)
    db.init_db()
    return db


@pytest.mark.unit
def test_execution_engine_updates_unrealized_pnl_on_bar() -> None:
    logger = StructuredLogger("test_unrealized", level="INFO")
    engine = ExecutionEngine(config={"index_symbol": "SPY"}, logger=logger, db=None)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    engine.market_state = MarketState(time=now, regime=Regime.CHOP)

    st = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=10),
        indicators={},
        position=Position(
            symbol="AAPL",
            side=Side.LONG,
            qty=2,
            avg_price=100.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            strategy="s",
        ),
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    engine.symbol_states["AAPL"] = st

    engine.on_bar(
        "AAPL",
        Bar(symbol="AAPL", time=now, open=0, high=0, low=0, close=105.0, volume=1),
    )
    assert st.position is not None
    assert st.position.unrealized_pnl == pytest.approx(10.0)


@pytest.mark.unit
def test_execution_engine_on_bar_accepts_single_bar_argument() -> None:
    logger = StructuredLogger("test_on_bar_arity", level="INFO")
    engine = ExecutionEngine(config={"index_symbol": "SPY"}, logger=logger, db=None)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    engine.market_state = MarketState(time=now, regime=Regime.CHOP)

    st = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=10),
        indicators={},
        position=Position(
            symbol="AAPL",
            side=Side.LONG,
            qty=2,
            avg_price=100.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            strategy="s",
        ),
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    engine.symbol_states["AAPL"] = st

    engine.on_bar(Bar(symbol="AAPL", time=now, open=0, high=0, low=0, close=105.0, volume=1))
    assert st.position is not None
    assert st.position.unrealized_pnl == pytest.approx(10.0)


@pytest.mark.unit
def test_trade_persistence_uses_closed_qty_not_fill_qty(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    logger = StructuredLogger("test_trade_qty", level="INFO")
    engine = ExecutionEngine(config={"index_symbol": "SPY"}, logger=logger, db=db)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    engine.market_state = MarketState(time=now, regime=Regime.CHOP)

    st = SymbolState(
        symbol="AAPL",
        bars=deque(
            [Bar(symbol="AAPL", time=now, open=0, high=0, low=0, close=100, volume=1)],
            maxlen=10,
        ),
        indicators={},
        position=Position(
            symbol="AAPL",
            side=Side.LONG,
            qty=1,
            avg_price=100.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            strategy="s",
            entry_time=now - timedelta(minutes=1),
            correlation_id="cid",
            open_risk=10.0,
        ),
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    engine.symbol_states["AAPL"] = st

    # Fill quantity exceeds open position; closed qty must be capped at 1.
    engine.on_fill(
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 2,
            "price": 101.0,
            "timestamp": now,
            "correlation_id": "cid",
        }
    )

    with db.get_session() as session:
        row = session.query(DbTrade).first()
        assert row is not None
        assert row.qty == pytest.approx(1.0)


@pytest.mark.unit
def test_trade_pnl_net_subtracts_configured_costs(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    logger = StructuredLogger("test_trade_costs", level="INFO")
    cfg = {
        "index_symbol": "SPY",
        "risk": {
            "commission_per_share": 0.01,
            "min_commission": 0.5,
            "slippage_bps": 10.0,
        },
    }
    engine = ExecutionEngine(config=cfg, logger=logger, db=db)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    engine.market_state = MarketState(time=now, regime=Regime.CHOP)

    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=10),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={
            "pending_entries": {
                "cid": {
                    "strategy": "s",
                    "entry_time": now,
                    "open_risk": 10.0,
                    "stop_price": 99.0,
                    "target_price": 102.0,
                }
            }
        },
    )

    # Open then close.
    engine.on_fill(
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "price": 100.0,
            "timestamp": now,
            "correlation_id": "cid",
        }
    )
    engine.on_fill(
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 10,
            "price": 101.0,
            "timestamp": now + timedelta(minutes=1),
            "correlation_id": "cid",
        }
    )

    with db.get_session() as session:
        row = session.query(DbTrade).first()
        assert row is not None
        assert row.pnl_gross == pytest.approx(10.0)
        assert row.pnl_net == pytest.approx(6.99)


@pytest.mark.unit
def test_fill_persistence_never_writes_none_correlation_id(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    logger = StructuredLogger("test_fill_corr_id", level="INFO")
    engine = ExecutionEngine(config={"index_symbol": "SPY"}, logger=logger, db=db)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    engine.market_state = MarketState(time=now, regime=Regime.CHOP)

    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=10),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    engine.on_fill(
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 1,
            "price": 100.0,
            "timestamp": now,
            "correlation_id": None,
            "broker_order_id": "oid-1",
        }
    )

    with db.get_session() as session:
        row = session.query(DbOrder).filter(DbOrder.broker_order_id == "oid-1").first()
        assert row is not None
        assert row.correlation_id != ""
        assert row.correlation_id != "None"
        assert row.correlation_id == "alpaca-oid-1"
