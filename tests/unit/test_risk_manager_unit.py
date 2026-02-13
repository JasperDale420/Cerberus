from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.domain import MarketState, OrderSide, Regime, Signal, SymbolState
from src.engine.risk import RiskManager


def _signal(**overrides):
    base: dict[str, Any] = dict(
        symbol="AAPL",
        side=OrderSide.BUY,
        size_hint=0.0,
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        strategy="s",
        regime=Regime.CHOP,
        generated_at=datetime.now(timezone.utc),
        meta={},
        correlation_id="c1",
    )
    base.update(overrides)
    return Signal(**base)


@pytest.mark.unit
def test_apply_rejects_when_daily_loss_exceeded() -> None:
    logger = MagicMock()
    rm = RiskManager({"max_daily_loss": 100.0, "max_risk_per_trade": 50.0}, logger)
    rm.current_daily_pnl = -101.0

    intents = rm.apply(
        _signal(),
        SymbolState("AAPL", deque(), {}, None, {}, [], {}),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
        current_positions=[],
    )
    assert intents == []


@pytest.mark.unit
def test_apply_rejects_when_stop_price_invalid() -> None:
    logger = MagicMock()
    rm = RiskManager({"max_risk_per_trade": 50.0}, logger)

    intents = rm.apply(
        _signal(stop_price=100.0),
        SymbolState("AAPL", deque(), {}, None, {}, [], {}),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
        current_positions=[],
    )
    assert intents == []


@pytest.mark.unit
def test_apply_rejects_when_notional_exceeds_limit() -> None:
    logger = MagicMock()
    rm = RiskManager({"max_risk_per_trade": 50.0, "max_notional_per_order": 10.0}, logger)

    intents = rm.apply(
        _signal(entry_price=100.0, stop_price=99.0),
        SymbolState("AAPL", deque(), {}, None, {}, [], {}),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
        current_positions=[],
    )
    assert intents == []


@pytest.mark.unit
def test_apply_respects_size_hint_but_caps_to_risk_limit() -> None:
    logger = MagicMock()
    rm = RiskManager({"max_risk_per_trade": 50.0, "max_notional_per_order": 1_000_000.0}, logger)

    # risk_per_share = 1 => qty_limit = 50, size_hint is higher => cap at 50
    intents = rm.apply(
        _signal(size_hint=999.0),
        SymbolState("AAPL", deque(), {}, None, {}, [], {}),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
        current_positions=[],
    )
    assert len(intents) == 1
    assert intents[0].qty == 50
    assert rm.daily_order_count == 1
