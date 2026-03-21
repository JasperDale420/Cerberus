from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

import pytest

from src.core.domain import (
    Bar,
    MarketState,
    OrderType,
    Position,
    Regime,
    Side,
    SymbolState,
)
from src.engine.position_manager import PositionManager


def _bar(symbol: str, t: datetime, *, o: float, h: float, low: float, c: float) -> Bar:
    return Bar(symbol=symbol, time=t, open=o, high=h, low=low, close=c, volume=0.0)


@pytest.mark.unit
def test_position_manager_prioritizes_target_when_stop_and_target_hit_same_bar() -> None:
    """M3 fix: When both stop and target trigger on same bar, target wins (trader-friendly)."""
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    pos = Position(
        symbol="AAPL",
        side=Side.LONG,
        qty=10,
        avg_price=100.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        strategy="s",
        entry_time=now,
        correlation_id="cid",
        open_risk=10.0,  # 1R per share
        stop_price=99.0,
        target_price=102.0,
    )
    state = SymbolState(
        symbol="AAPL",
        bars=deque([_bar("AAPL", now, o=100, h=103, low=98, c=101)], maxlen=10),
        indicators={},
        position=pos,
        open_orders={},
        allowed_strategies=["s"],
        meta={},
    )
    market = MarketState(time=now, regime=Regime.CHOP)

    decision = PositionManager().on_bar(state, market)
    # M3: Target takes priority over stop when both hit (more favorable for trader)
    assert decision.reason == "TARGET_HIT"
    assert decision.intent is not None
    assert decision.intent.order_type == OrderType.MARKET
    assert decision.intent.meta.get("exit_reason") == "TARGET_HIT"


@pytest.mark.unit
def test_position_manager_max_hold_exits_before_stop_target_checks() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    pos = Position(
        symbol="AAPL",
        side=Side.LONG,
        qty=10,
        avg_price=100.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        strategy="s",
        entry_time=now - timedelta(seconds=20),
        correlation_id="cid",
        open_risk=10.0,
        stop_price=99.0,
        target_price=102.0,
        max_hold_seconds=10,
    )
    state = SymbolState(
        symbol="AAPL",
        bars=deque([_bar("AAPL", now, o=100, h=101, low=99.5, c=100)], maxlen=10),
        indicators={},
        position=pos,
        open_orders={},
        allowed_strategies=["s"],
        meta={},
    )
    market = MarketState(time=now, regime=Regime.CHOP)

    decision = PositionManager().on_bar(state, market)
    assert decision.reason == "MAX_HOLD_EXCEEDED"
    assert decision.intent is not None
    assert decision.intent.meta.get("exit_reason") == "MAX_HOLD_EXCEEDED"


@pytest.mark.unit
def test_position_manager_updates_mae_mfe_in_r_units() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    pos = Position(
        symbol="AAPL",
        side=Side.LONG,
        qty=10,
        avg_price=100.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        strategy="s",
        entry_time=now,
        correlation_id="cid",
        open_risk=10.0,  # 1R per share
        stop_price=None,
        target_price=None,
    )
    state = SymbolState(
        symbol="AAPL",
        bars=deque([_bar("AAPL", now, o=100, h=102, low=98, c=101)], maxlen=10),
        indicators={},
        position=pos,
        open_orders={},
        allowed_strategies=["s"],
        meta={},
    )
    market = MarketState(time=now, regime=Regime.CHOP)

    decision = PositionManager().on_bar(state, market)
    assert decision.intent is None
    assert pos.mae_r == pytest.approx(2.0)  # low 98 is -2R from 100 with 1R per share
    assert pos.mfe_r == pytest.approx(2.0)  # high 102 is +2R


@pytest.mark.unit
def test_position_manager_on_fill_opens_position_from_pending_entry_context() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    state = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=10),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={
            "pending_entries": {
                "cid": {
                    "strategy": "vwap_reversion",
                    "open_risk": 10.0,
                    "stop_price": 99.0,
                    "target_price": 102.0,
                    "features": {"scanner_score": 0.9},
                    "entry_time": now,
                    "max_hold_seconds": 60,
                }
            }
        },
    )
    market = MarketState(time=now, regime=Regime.CHOP)

    decision = PositionManager().on_fill(
        state,
        market,
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 2,
            "price": 100.0,
            "timestamp": now,
            "correlation_id": "cid",
        },
        risk_cfg={
            "commission_per_share": 0.01,
            "min_commission": 0.0,
            "slippage_bps": 0.0,
        },
    )

    assert decision.event == "opened"
    assert state.position is not None
    assert state.position.strategy == "vwap_reversion"
    # open_risk is recalculated from actual fill price and stop: abs(100-99)*2 = 2.0
    assert state.position.open_risk == pytest.approx(2.0)
    assert state.position.stop_price == pytest.approx(99.0)
    assert state.position.target_price == pytest.approx(102.0)
    assert state.position.correlation_id == "cid"
    assert state.position.commission == pytest.approx(0.02)


@pytest.mark.unit
def test_position_manager_on_fill_closes_position_and_returns_closed_trade_info() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    state = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=10),
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
            commission=0.01,
            slippage_estimate=0.0,
        ),
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    market = MarketState(time=now, regime=Regime.BULL)

    decision = PositionManager().on_fill(
        state,
        market,
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 2,  # more than open; should cap close_qty at 1
            "price": 101.0,
            "timestamp": now,
            "correlation_id": "cid",
        },
        risk_cfg={
            "commission_per_share": 0.0,
            "min_commission": 0.0,
            "slippage_bps": 0.0,
        },
    )

    assert decision.event == "closed"
    assert state.position is None
    assert decision.closed_trade is not None
    assert decision.closed_trade.qty == pytest.approx(1.0)
    assert decision.closed_trade.pnl_gross == pytest.approx(1.0)
    assert decision.closed_trade.pnl_r == pytest.approx(0.1)


@pytest.mark.unit
def test_position_manager_skips_stop_target_when_broker_managed_exits_enabled() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    pos = Position(
        symbol="AAPL",
        side=Side.LONG,
        qty=10,
        avg_price=100.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        strategy="s",
        entry_time=now,
        correlation_id="cid",
        open_risk=10.0,
        stop_price=99.0,
        target_price=102.0,
    )
    state = SymbolState(
        symbol="AAPL",
        bars=deque([_bar("AAPL", now, o=100, h=103, low=98, c=101)], maxlen=10),
        indicators={},
        position=pos,
        open_orders={},
        allowed_strategies=["s"],
        meta={},
    )
    market = MarketState(time=now, regime=Regime.CHOP)

    decision = PositionManager().on_bar(state, market, broker_managed_exits=True)
    assert decision.intent is None
    assert decision.reason is None
