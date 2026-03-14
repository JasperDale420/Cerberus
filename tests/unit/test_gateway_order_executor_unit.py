from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.core.domain import OrderIntent, OrderSide, OrderType
from src.engine.orders import GatewayOrderExecutor


def test_gateway_order_executor_submits_via_central_api_and_persists_order() -> None:
    logger = MagicMock()
    db = MagicMock()
    gateway = MagicMock()
    gateway.submit_order.return_value = {
        "id": "gw-order-1",
        "symbol": "AAPL",
        "status": "accepted",
    }
    fixed_now = datetime(2026, 2, 13, tzinfo=timezone.utc)
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=5.0,
        order_type=OrderType.MARKET,
        limit_price=None,
        time_in_force="day",
        correlation_id="corr-123",
        strategy="vwap_reversion",
        stop_loss=190.0,
        take_profit=210.0,
        meta={"created_at": fixed_now.isoformat()},
    )

    executor = GatewayOrderExecutor(gateway, logger, db=db, clock=lambda: fixed_now)

    out = executor.submit(intent)

    assert out["id"] == "gw-order-1"
    gateway.submit_order.assert_called_once()
    args = gateway.submit_order.call_args.kwargs
    assert args["symbol"] == "AAPL"
    assert args["side"] == "buy"
    assert args["qty"] == 5.0
    assert args["order_type"] == "market"
    assert args["client_order_id"] == "corr-123"

    db.write.assert_called_once()
    operation, callback = db.write.call_args.args
    assert operation == "order"
    session = MagicMock()
    callback(session)
    assert session.add.call_count == 1
    row = session.add.call_args.args[0]
    assert row.broker_order_id == "gw-order-1"
    assert row.symbol == "AAPL"
    assert row.meta_json["strategy"] == "vwap_reversion"


def test_gateway_order_executor_ignores_broker_managed_exit_fields() -> None:
    logger = MagicMock()
    db = MagicMock()
    gateway = MagicMock()
    gateway.submit_order.return_value = {"id": "gw-order-2"}
    intent = OrderIntent(
        symbol="TSLA",
        side=OrderSide.SELL,
        qty=2.0,
        order_type=OrderType.LIMIT,
        limit_price=250.0,
        time_in_force="day",
        correlation_id="corr-456",
        strategy="orb",
        stop_loss=255.0,
        take_profit=240.0,
        meta={},
    )
    executor = GatewayOrderExecutor(gateway, logger, db=db, clock=lambda: datetime.now(timezone.utc))
    executor.broker_managed_exits = False

    executor.submit(intent)

    call = gateway.submit_order.call_args.kwargs
    assert call["symbol"] == "TSLA"
    assert call["order_type"] == "limit"
    assert call["limit_price"] == 250.0
    assert "stop_price" not in call
