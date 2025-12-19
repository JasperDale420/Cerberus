from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.domain import OrderIntent, OrderSide, OrderType
from src.core.logger import StructuredLogger
from src.engine.orders import OrderExecutor


@dataclass
class _EnumLike:
    value: str


@pytest.mark.unit
def test_order_executor_handle_trade_update_normalizes_fill_event() -> None:
    logger = StructuredLogger("test_trade_update", level="INFO")
    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    executor = OrderExecutor(
        alpaca, logger, db=None, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )

    order = SimpleNamespace(
        id="oid-1",
        symbol="AAPL",
        client_order_id="corr-1",
        side=_EnumLike("buy"),
        status=_EnumLike("filled"),
    )
    update = SimpleNamespace(
        event=_EnumLike("fill"),
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        order=order,
        qty=3,
        price=101.25,
    )

    out = executor.handle_trade_update(update)
    assert out["event"] == "fill"
    assert out["symbol"] == "AAPL"
    assert out["side"] == "buy"
    assert out["status"] == "filled"
    assert out["broker_order_id"] == "oid-1"
    assert out["client_order_id"] == "corr-1"
    assert out["correlation_id"] == "corr-1"
    assert out["fill_qty"] == pytest.approx(3.0)
    assert out["fill_price"] == pytest.approx(101.25)


@pytest.mark.unit
def test_order_executor_handle_trade_update_falls_back_correlation_id_when_missing() -> (
    None
):
    logger = StructuredLogger("test_trade_update_fallback", level="INFO")
    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    executor = OrderExecutor(
        alpaca, logger, db=None, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )

    order = SimpleNamespace(
        id="oid-1",
        symbol="AAPL",
        client_order_id="",
        side=_EnumLike("buy"),
        status=_EnumLike("filled"),
    )
    update = SimpleNamespace(
        event=_EnumLike("fill"),
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        order=order,
        qty=3,
        price=101.25,
    )

    out = executor.handle_trade_update(update)
    assert out["broker_order_id"] == "oid-1"
    assert not out["client_order_id"]
    assert out["correlation_id"] == "alpaca-oid-1"


@pytest.mark.unit
def test_order_executor_cancel_all_for_symbol_calls_cancel_by_id() -> None:
    logger = StructuredLogger("test_cancel_all_for_symbol", level="INFO")
    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    alpaca.trading_client.get_orders.return_value = [
        SimpleNamespace(id="oid-1", symbol="AAPL"),
        SimpleNamespace(id="oid-2", symbol="MSFT"),
    ]

    executor = OrderExecutor(alpaca, logger)
    executor.cancel_all_for_symbol("AAPL")

    alpaca.trading_client.cancel_order_by_id.assert_called_once_with("oid-1")


@pytest.mark.unit
def test_order_executor_submit_maps_time_in_force_variants() -> None:
    logger = StructuredLogger("test_tif_mapping", level="INFO")
    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    alpaca.trading_client.submit_order.return_value = SimpleNamespace(id="oid-1")

    executor = OrderExecutor(alpaca, logger, db=None)
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=1,
        order_type=OrderType.MARKET,
        limit_price=None,
        time_in_force="gtc",
        correlation_id="corr-1",
        strategy="s",
        stop_loss=None,
        take_profit=None,
        meta={},
    )
    executor.submit(intent)
    req = alpaca.trading_client.submit_order.call_args[0][0]
    assert str(req.time_in_force.value).lower() == "gtc"
