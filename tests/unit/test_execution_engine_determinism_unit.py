from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.core.domain import OrderSide, Regime, Signal, SymbolState
from src.engine.execution import ExecutionEngine


@pytest.mark.unit
def test_apply_scan_result_is_deterministic_for_same_inputs() -> None:
    logger = MagicMock()
    engine = ExecutionEngine(
        config={"index_symbol": "SPY", "max_churn_per_scan": 1},
        logger=logger,
        alpaca_client=None,
        db=None,
    )

    # Existing tracked symbols.
    engine.symbol_states["SPY"] = SymbolState(
        symbol="SPY",
        bars=deque(maxlen=100),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    engine.symbol_states["AAA"] = SymbolState(
        symbol="AAA",
        bars=deque(maxlen=100),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    engine.symbol_states["BBB"] = SymbolState(
        symbol="BBB",
        bars=deque(maxlen=100),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    # Scanner wants to keep BBB, drop AAA, add CCC, in this order.
    scan_results = [
        MagicMock(symbol="BBB", strategies=["s1"], score=1.0, features=None),
        MagicMock(symbol="CCC", strategies=["s2"], score=2.0, features=None),
    ]

    # Run twice from identical starting state by reinitializing engine.
    def run_once() -> tuple[set[str], list[tuple]]:
        e = ExecutionEngine(
            config={"index_symbol": "SPY", "max_churn_per_scan": 1},
            logger=logger,
            alpaca_client=None,
            db=None,
        )
        e.symbol_states = dict(engine.symbol_states)
        e.apply_scan_result(scan_results)  # type: ignore
        return set(e.symbol_states.keys()), list(logger.info.call_args_list)

    keys1, logs1 = run_once()
    logger.info.reset_mock()
    keys2, logs2 = run_once()

    assert keys1 == keys2
    assert logs1 == logs2


@pytest.mark.unit
def test_process_signal_degrades_gracefully_on_risk_exception() -> None:
    logger = MagicMock()
    engine = ExecutionEngine(
        config={"index_symbol": "SPY"},
        logger=logger,
        alpaca_client=None,
        db=None,
    )
    engine.market_state.regime = Regime.CHOP

    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=100),
        position=None,
        indicators={},
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    engine.risk_manager.apply = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore

    t0 = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
    sig = Signal(
        symbol="AAPL",
        side=OrderSide.BUY,
        size_hint=0.0,
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        strategy="s",
        regime=Regime.CHOP,
        generated_at=t0,
        meta={},
        correlation_id="corr-1",
    )

    engine._process_signal(sig)
    assert engine.error_counts["risk"] == 1
