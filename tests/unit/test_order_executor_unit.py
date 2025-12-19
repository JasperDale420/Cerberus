from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.core.config import ConfigLoader
from src.core.domain import OrderIntent, OrderSide, OrderType
from src.core.logger import StructuredLogger
from src.engine.orders import OrderExecutor


@pytest.mark.unit
def test_order_executor_submits_order_and_persists_db_row(tmp_path: Path) -> None:
    # DB
    db_path = tmp_path / "orders.db"
    db_url = f"sqlite:///{db_path}"
    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url}
    logger = StructuredLogger("TestOrderExec", level="INFO")

    db = DatabaseDatabase(loader, logger)
    db.init_db()

    # Alpaca stub
    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    alpaca.trading_client.submit_order.return_value = MagicMock(id="order-1")

    executor = OrderExecutor(alpaca, logger, db=db)
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        time_in_force="day",
        correlation_id="corr-1",
        strategy="s",
        stop_loss=99.0,
        take_profit=102.0,
        meta={"k": "v"},
    )

    out = executor.submit(intent)
    assert out is not None
    alpaca.trading_client.submit_order.assert_called_once()
    req = alpaca.trading_client.submit_order.call_args[0][0]
    assert req.time_in_force is not None
    assert str(req.time_in_force.value).lower() == "day"
    # PRD 6.5: broker-managed OCO exits via bracket order fields.
    assert getattr(req, "take_profit", None) is not None
    assert float(getattr(req.take_profit, "limit_price", 0.0) or 0.0) == 102.0
    assert getattr(req, "stop_loss", None) is not None
    assert float(getattr(req.stop_loss, "stop_price", 0.0) or 0.0) == 99.0

    with db.get_session() as session:
        orders = session.query(DbOrder).all()
        assert len(orders) == 1
        assert orders[0].correlation_id == "corr-1"
        assert orders[0].symbol == "AAPL"
        assert orders[0].status == "submitted"
        assert orders[0].meta_json is not None
        assert orders[0].meta_json["k"] == "v"
