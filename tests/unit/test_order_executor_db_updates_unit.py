from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.core.config import ConfigLoader
from src.core.domain import OrderIntent, OrderSide, OrderType
from src.core.logger import StructuredLogger
from src.engine.orders import OrderExecutor


def _make_db(tmp_path: Path) -> DatabaseDatabase:
    db_path = tmp_path / "orders_updates.db"
    db_url = f"sqlite:///{db_path}"
    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url}
    logger = StructuredLogger("TestOrderDB", level="INFO")
    db = DatabaseDatabase(loader, logger)
    db.init_db()
    return db


def _intent() -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=1,
        order_type=OrderType.MARKET,
        limit_price=None,
        time_in_force="day",
        correlation_id="corr-1",
        strategy="s",
        stop_loss=99.0,
        take_profit=102.0,
        meta={},
    )


@pytest.mark.unit
def test_order_executor_persists_failed_order_on_submit_exception(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    logger = StructuredLogger("TestOrderFail", level="INFO")

    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    alpaca.trading_client.submit_order.side_effect = RuntimeError("no route")

    ex = OrderExecutor(alpaca, logger, db=db, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(RuntimeError, match="no route"):
        ex.submit(_intent())

    with db.get_session() as session:
        rows = session.query(DbOrder).all()
        assert len(rows) == 1
        assert rows[0].status == "order_failed"
        assert rows[0].meta_json is not None
        assert rows[0].meta_json["correlation_id"] == "corr-1"
        assert "error" in rows[0].meta_json
        assert "broker_error_payload" in rows[0].meta_json
        assert rows[0].meta_json["broker_error_payload"]["exception_type"] == "RuntimeError"


@pytest.mark.unit
def test_order_executor_persists_broker_response_payload_with_truncation(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    logger = StructuredLogger("TestOrderFailPayload", level="INFO")

    class _Resp:
        status_code = 418

        def text(self) -> str:
            return "x" * 5000

    class _BrokerError(RuntimeError):
        def __init__(self, msg: str):
            super().__init__(msg)
            self.status_code = 418
            self.request_id = "req-1"
            self.response = _Resp()

    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    alpaca.trading_client.submit_order.side_effect = _BrokerError("teapot")

    ex = OrderExecutor(alpaca, logger, db=db, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(_BrokerError, match="teapot"):
        ex.submit(_intent())

    with db.get_session() as session:
        row = session.query(DbOrder).first()
        assert row is not None
        assert row.meta_json is not None
        payload = row.meta_json.get("broker_error_payload")
        assert isinstance(payload, dict)
        assert payload.get("status_code") == 418
        assert payload.get("request_id") == "req-1"
        assert payload.get("response_status_code") == 418
        text = payload.get("response_text")
        assert isinstance(text, str)
        assert text.endswith("...(truncated)")
        assert len(text) <= 2015


@pytest.mark.unit
def test_order_executor_trade_update_updates_existing_db_row(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    logger = StructuredLogger("TestOrderUpdate", level="INFO")

    with db.get_session() as session:
        session.add(
            DbOrder(
                correlation_id="corr-1",
                symbol="AAPL",
                side="buy",
                qty=1,
                type="market",
                limit_price=None,
                status="submitted",
                time_placed=datetime(2025, 1, 1, tzinfo=timezone.utc),
                time_last_update=datetime(2025, 1, 1, tzinfo=timezone.utc),
                broker_order_id="oid-1",
                meta_json={"strategy": "s"},
            )
        )
        session.commit()

    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    ex = OrderExecutor(alpaca, logger, db=db, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))

    order = SimpleNamespace(
        id="oid-1",
        symbol="AAPL",
        client_order_id="",
        side=SimpleNamespace(value="buy"),
        status=SimpleNamespace(value="filled"),
        type=SimpleNamespace(value="market"),
        qty="1",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    update = SimpleNamespace(
        event=SimpleNamespace(value="fill"),
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        order=order,
        qty=1,
        price=101.0,
    )

    out = ex.handle_trade_update(update)
    assert out["correlation_id"] == "corr-1"
    assert out["strategy"] == "s"
    assert out["status"] == "filled"

    with db.get_session() as session:
        row = session.query(DbOrder).filter(DbOrder.broker_order_id == "oid-1").first()
        assert row is not None
        assert row.status == "filled"
        assert row.meta_json is not None
        assert row.meta_json.get("reconciled") is not True  # trade update writes broker_status fields


@pytest.mark.unit
def test_order_executor_trade_update_upserts_when_missing_row(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    logger = StructuredLogger("TestOrderUpsert", level="INFO")
    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    ex = OrderExecutor(alpaca, logger, db=db, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))

    order = SimpleNamespace(
        id="oid-2",
        symbol="MSFT",
        client_order_id="corr-2",
        side=SimpleNamespace(value="sell"),
        status=SimpleNamespace(value="new"),
        type=SimpleNamespace(value="limit"),
        qty="2",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        limit_price=300.0,
    )
    update = SimpleNamespace(
        event=SimpleNamespace(value="new"),
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        order=order,
    )

    out = ex.handle_trade_update(update)
    assert out["correlation_id"] == "corr-2"
    assert out["strategy"] is None

    with db.get_session() as session:
        row = session.query(DbOrder).filter(DbOrder.broker_order_id == "oid-2").first()
        assert row is not None
        assert row.correlation_id == "corr-2"
        assert row.meta_json is not None
        assert row.meta_json.get("recovered_from_trade_stream") is True


@pytest.mark.unit
def test_order_executor_trade_update_inherits_correlation_id_from_parent_order(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    logger = StructuredLogger("TestOrderUpsertParent", level="INFO")

    with db.get_session() as session:
        session.add(
            DbOrder(
                correlation_id="corr-parent",
                symbol="AAPL",
                side="buy",
                qty=1,
                type="market",
                limit_price=None,
                status="submitted",
                time_placed=datetime(2025, 1, 1, tzinfo=timezone.utc),
                time_last_update=datetime(2025, 1, 1, tzinfo=timezone.utc),
                broker_order_id="oid-parent",
                meta_json={"strategy": "s"},
            )
        )
        session.commit()

    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    ex = OrderExecutor(alpaca, logger, db=db, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))

    # Simulate a bracket/OTO child order update without client_order_id.
    order = SimpleNamespace(
        id="oid-child",
        symbol="AAPL",
        client_order_id="",
        parent_order_id="oid-parent",
        side=SimpleNamespace(value="sell"),
        status=SimpleNamespace(value="new"),
        type=SimpleNamespace(value="limit"),
        qty="1",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        limit_price=100.0,
    )
    update = SimpleNamespace(
        event=SimpleNamespace(value="new"),
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        order=order,
    )

    out = ex.handle_trade_update(update)
    assert out["correlation_id"] == "corr-parent"
    assert out["strategy"] == "s"
    assert out["parent_broker_order_id"] == "oid-parent"

    with db.get_session() as session:
        row = session.query(DbOrder).filter(DbOrder.broker_order_id == "oid-child").first()
        assert row is not None
        assert row.correlation_id == "corr-parent"
        assert row.meta_json is not None
        assert row.meta_json.get("parent_broker_order_id") == "oid-parent"
