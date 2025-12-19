from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.core.config import ConfigLoader
from src.core.domain import OrderIntent, OrderSide, OrderType
from src.core.logger import StructuredLogger
from src.engine.orders import NoopOrderExecutor


@pytest.mark.unit
def test_noop_order_executor_persists_simulated_order(tmp_path: Path) -> None:
    db_path = tmp_path / "noop.db"
    db_url = f"sqlite:///{db_path}"

    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url}
    logger = StructuredLogger("NoopExecTest", level="INFO")

    db = DatabaseDatabase(loader, logger)
    db.init_db()

    fixed_now = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

    exec_ = NoopOrderExecutor(logger, db=db, clock=lambda: fixed_now)
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=1.0,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        time_in_force="day",
        correlation_id="corr-1",
        strategy="oneshot",
        stop_loss=99.0,
        take_profit=102.0,
        meta={"created_at": fixed_now.isoformat()},
    )

    out = exec_.submit(intent)
    assert out["id"] == "noop-corr-1"
    assert out["status"] == "simulated"

    with db.get_session() as session:
        row = session.query(DbOrder).one()
        assert row.correlation_id == "corr-1"
        assert row.status == "simulated"
        assert row.broker_order_id == "noop-corr-1"
        assert (row.meta_json or {}).get("simulated") is True
