from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.analysis.schema import Signal as DbSignal
from src.core.config import ConfigLoader
from src.core.domain import Bar, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.engine.execution import ExecutionEngine
from src.strategies.base import BaseStrategy


class _OneShotStrategy(BaseStrategy):
    name = "oneshot"

    def on_bar(self, symbol: str, bar: Bar, symbol_state: SymbolState, market_state):
        if symbol_state.meta.get("fired"):
            return None
        symbol_state.meta["fired"] = True
        return Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            size_hint=0.0,
            entry_price=bar.close,
            stop_price=bar.close - 1.0,
            target_price=bar.close + 2.0,
            strategy=self.name,
            regime=market_state.regime,
            generated_at=bar.time,
            meta={"source": "e2e"},
            correlation_id="corr-1",
        )


@pytest.mark.e2e
def test_offline_signal_to_order_and_db_persistence(tmp_path: Path) -> None:
    # DB
    db_path = tmp_path / "smoke.db"
    db_url = f"sqlite:///{db_path}"
    loader = MagicMock(spec=ConfigLoader)
    loader.load_config.return_value = {"database_url": db_url}
    logger = StructuredLogger("E2ESmoke", level="INFO")

    db = DatabaseDatabase(loader, logger)
    db.init_db()

    # Broker stub
    alpaca = MagicMock()
    alpaca.trading_client = MagicMock()
    alpaca.trading_client.submit_order.return_value = MagicMock(id="order-1")

    # Engine
    config = {
        "index_symbol": "SPY",
        "max_daily_loss": 1_000_000.0,
        "max_risk_per_trade": 50.0,
        "max_notional_per_order": 1_000_000.0,
    }
    engine = ExecutionEngine(config=config, logger=logger, db=db, alpaca_client=alpaca)
    engine.market_state.regime = Regime.CHOP

    engine.register_strategy(_OneShotStrategy({}, logger))

    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=100),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=["oneshot"],
        meta={},
    )

    t0 = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
    engine.on_bar(
        "AAPL",
        Bar(
            symbol="AAPL",
            time=t0,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000.0,
        ),
    )

    # Order submission happened via OrderExecutor.
    alpaca.trading_client.submit_order.assert_called_once()

    with db.get_session() as session:
        signals = session.query(DbSignal).all()
        orders = session.query(DbOrder).all()

        assert len(signals) == 1
        assert signals[0].correlation_id == "corr-1"
        assert signals[0].accepted is True

        assert len(orders) == 1
        assert orders[0].correlation_id == "corr-1"
        assert orders[0].status == "submitted"
