from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, List, Optional

import pytest

from src.core.domain import (
    Bar,
    MarketState,
    Position,
    Regime,
    RiskMode,
    Side,
    SymbolFeatures,
    SymbolState,
    WatchlistSymbol,
)
from src.core.logger import StructuredLogger
from src.engine.execution import ExecutionEngine


def _logger(name: str) -> StructuredLogger:
    return StructuredLogger(name, level="INFO")


@pytest.mark.unit
def test_execution_engine_routing_excludes_disabled_strategy_regime_pair() -> None:
    from src.core.domain import OrderSide, Signal
    from src.strategies.base import BaseStrategy

    class _S(BaseStrategy):
        name = "s1"

        def __init__(self, config: dict, logger: StructuredLogger):
            super().__init__(config, logger)
            self.calls = 0

        def on_bar(
            self,
            symbol: str,
            bar: Bar,
            symbol_state: SymbolState,
            market_state: MarketState,
        ):
            self.calls += 1
            return Signal(
                symbol=symbol,
                side=OrderSide.BUY,
                size_hint=1.0,
                entry_price=float(bar.close),
                stop_price=float(bar.close) - 1.0,
                target_price=float(bar.close) + 1.0,
                strategy=self.name,
                regime=market_state.regime,
                generated_at=market_state.time,
                meta={},
            )

    alp = _FakeAlpacaClient()
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    cfg = {
        "index_symbol": "SPY",
        "strategy_routing": {"bull": ["s1"], "bear": [], "chop": []},
        "strategies": {"s1": {"regimes": {"bull": {"enabled": False}}}},
    }
    engine = ExecutionEngine(
        config=cfg,
        logger=_logger("test_exec_disabled_routing"),
        alpaca_client=alp,  # type: ignore
        clock=lambda: now,
    )
    strat = _S({}, _logger("test_exec_disabled_routing_strat"))
    engine.register_strategy(strat)

    engine.market_state = MarketState(
        time=now, regime=Regime.BULL, risk_mode=RiskMode.NORMAL
    )

    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(
            [Bar(symbol="AAPL", time=now, open=1, high=1, low=1, close=1, volume=1)],
            maxlen=100,
        ),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=["s1"],
        meta={},
    )
    engine.on_bar(
        "AAPL",
        Bar(symbol="AAPL", time=now, open=1, high=2, low=0.5, close=1.5, volume=10),
    )

    assert strat.calls == 0


class _FakeTradingClient:
    def __init__(self) -> None:
        self.cancel_orders_calls = 0
        self.close_all_positions_calls = 0
        self._positions: List[Any] = []
        self._orders: List[Any] = []
        self.close_raises: Optional[Exception] = None

    def cancel_orders(self) -> None:
        self.cancel_orders_calls += 1

    def close_all_positions(self, *_args: Any, **_kwargs: Any) -> None:
        self.close_all_positions_calls += 1
        if self.close_raises is not None:
            raise self.close_raises

    def get_all_positions(self) -> List[Any]:
        return list(self._positions)

    def get_orders(self, *_args: Any, **_kwargs: Any) -> List[Any]:
        return list(self._orders)


class _FakeAlpacaClient:
    def __init__(self) -> None:
        self.trading_client = _FakeTradingClient()
        self.subscribed: List[str] = []
        self.unsubscribed: List[str] = []

    def subscribe(self, symbol: str) -> None:
        self.subscribed.append(symbol)

    def unsubscribe(self, symbol: str) -> None:
        self.unsubscribed.append(symbol)


def _features(symbol: str) -> SymbolFeatures:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return SymbolFeatures(
        symbol=symbol,
        price=1.0,
        atr_pct=0.0,
        avg_volume=0.0,
        intraday_range_pct=0.0,
        gap_pct=0.0,
        ema20_slope=0.0,
        ema_trend_strength=0.0,
        distance_from_vwap=0.0,
        premarket_volume=0.0,
        adx=0.0,
        distance_from_ema20=0.0,
        prior_day_high=0.0,
        prior_day_low=0.0,
        bb_upper=0.0,
        bb_lower=0.0,
        price_zscore=0.0,
        flow_zscore=0.0,
        call_put_ratio=0.0,
        large_sweeps_count=0,
        aggressive_flow_share=0.0,
        last_updated=now,
        extra={},
    )


def _watch(symbol: str, strategies: Optional[List[str]] = None) -> WatchlistSymbol:
    return WatchlistSymbol(
        symbol=symbol,
        score=1.0,
        strategies=list(strategies or ["s1"]),
        features=_features(symbol),
    )


@pytest.mark.unit
def test_execution_engine_update_config_syncs_risk_mode_and_limits() -> None:
    alp = _FakeAlpacaClient()
    engine = ExecutionEngine(
        config={"risk": {"risk_mode": "normal", "max_daily_loss": 100}},
        logger=_logger("test_exec_update_config"),
        alpaca_client=alp,  # type: ignore
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    engine.market_state = MarketState(
        time=engine.clock(), regime=Regime.CHOP, risk_mode=RiskMode.NORMAL
    )

    engine.update_config(
        {
            "risk": {
                "risk_mode": "off",
                "max_daily_loss": 123,
                "max_risk_per_trade": 1.5,
            },
            "max_open_risk": 9.0,
            "max_notional_per_symbol": 2000.0,
        }
    )

    assert engine.risk_manager.risk_mode == "off"
    assert engine.market_state.risk_mode == RiskMode.OFF
    assert engine.risk_manager.max_daily_loss == pytest.approx(123.0)
    assert engine.risk_manager.max_risk_per_trade == pytest.approx(1.5)


@pytest.mark.unit
def test_execution_engine_apply_scan_result_clamps_and_throttles_churn() -> None:
    alp = _FakeAlpacaClient()
    engine = ExecutionEngine(
        config={"index_symbol": "SPY", "max_churn_per_scan": 1},
        logger=_logger("test_exec_apply_scan"),
        alpaca_client=alp,  # type: ignore
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    engine.symbol_states["SPY"] = SymbolState(
        symbol="SPY",
        bars=deque(
            [Bar(symbol="SPY", time=now, open=1, high=1, low=1, close=1, volume=0)],
            maxlen=10,
        ),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )
    # Two old symbols; only one removal allowed per scan.
    engine.symbol_states["OLD1"] = replace(engine.symbol_states["SPY"], symbol="OLD1")
    engine.symbol_states["OLD2"] = replace(engine.symbol_states["SPY"], symbol="OLD2")
    # A symbol with an open position should be kept even if removed.
    keep = replace(engine.symbol_states["SPY"], symbol="KEEP")
    keep.position = Position(
        symbol="KEEP",
        side=Side.LONG,
        qty=1,
        avg_price=1.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        strategy="s",
    )
    engine.symbol_states["KEEP"] = keep

    # 31 requested => clamped to 29 because we reserve 1 slot for index_symbol (PRD ~30 WS limit).
    desired = [_watch(f"S{i}") for i in range(31)]
    # Also retain KEEP and add a new symbol.
    desired[0] = _watch("KEEP")
    desired[1] = _watch("NEW1")

    engine.apply_scan_result(desired)

    assert "KEEP" in engine.symbol_states
    assert "S29" not in engine.symbol_states
    assert "S30" not in engine.symbol_states
    # Churn budget=1: should remove exactly one from {OLD1, OLD2} and not add new ones if budget exhausted.
    removed = [s for s in ("OLD1", "OLD2") if s not in engine.symbol_states]
    assert len(removed) == 1


@pytest.mark.unit
def test_execution_engine_apply_scan_result_reserves_index_slot_for_ws_limit() -> None:
    alp = _FakeAlpacaClient()
    engine = ExecutionEngine(
        config={"index_symbol": "SPY", "max_churn_per_scan": 100},
        logger=_logger("test_exec_scan_reserve_index"),
        alpaca_client=alp,  # type: ignore
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    engine.symbol_states["SPY"] = SymbolState(
        symbol="SPY",
        bars=deque(
            [Bar(symbol="SPY", time=now, open=1, high=1, low=1, close=1, volume=0)],
            maxlen=10,
        ),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )

    desired = [_watch(f"S{i}") for i in range(31)]
    engine.apply_scan_result(desired)

    non_index = [s for s in engine.symbol_states.keys() if s != "SPY"]
    assert len(non_index) == 29
    assert "S29" not in engine.symbol_states
    assert "S30" not in engine.symbol_states
    assert len(alp.subscribed) == 29


@pytest.mark.unit
def test_execution_engine_flatten_all_resets_local_state_on_success() -> None:
    alp = _FakeAlpacaClient()
    engine = ExecutionEngine(
        config={"position_mismatch_mode": "ignore"},
        logger=_logger("test_exec_flatten_success"),
        alpaca_client=alp,  # type: ignore
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    engine.symbol_states["AAPL"] = SymbolState(
        symbol="AAPL",
        bars=deque(maxlen=10),
        indicators={},
        position=Position(
            symbol="AAPL",
            side=Side.LONG,
            qty=1,
            avg_price=1.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            strategy="s",
        ),
        open_orders={"o1": {"id": "o1"}},
        allowed_strategies=[],
        meta={},
    )

    engine.flatten_all(reason="unit_test")

    assert alp.trading_client.cancel_orders_calls == 1
    assert alp.trading_client.close_all_positions_calls == 1
    assert engine.symbol_states["AAPL"].position is None
    assert engine.symbol_states["AAPL"].open_orders == {}


@pytest.mark.unit
def test_execution_engine_flatten_all_raises_when_close_fails_in_halt_mode() -> None:
    alp = _FakeAlpacaClient()
    alp.trading_client.close_raises = RuntimeError("broker down")
    engine = ExecutionEngine(
        config={"position_mismatch_mode": "halt"},
        logger=_logger("test_exec_flatten_raise"),
        alpaca_client=alp,  # type: ignore
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(RuntimeError, match="broker down"):
        engine.flatten_all(reason="unit_test")


@pytest.mark.unit
def test_execution_engine_tracks_session_vwap_and_resets_on_day_boundary() -> None:
    engine = ExecutionEngine(
        config={"index_symbol": "SPY"},
        logger=_logger("test_exec_session_vwap"),
        alpaca_client=None,
        db=None,
    )

    t0 = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
    b1 = Bar(symbol="AAPL", time=t0, open=10, high=10, low=10, close=10, volume=100)
    engine.on_bar(b1)
    st = engine.symbol_states["AAPL"]
    assert st.indicators["session_day"] == t0.date()
    assert st.indicators["cum_vol"] == pytest.approx(100.0)
    assert st.indicators["session_vwap"] == pytest.approx(10.0)
    assert b1.vwap == pytest.approx(10.0)

    # Another bar same day updates cumulative VWAP deterministically.
    b2 = Bar(
        symbol="AAPL",
        time=t0.replace(minute=31),
        open=12,
        high=12,
        low=12,
        close=12,
        volume=100,
    )
    engine.on_bar(b2)
    st = engine.symbol_states["AAPL"]
    assert st.indicators["cum_vol"] == pytest.approx(200.0)
    assert st.indicators["session_vwap"] == pytest.approx(11.0)
    assert b2.vwap == pytest.approx(11.0)

    # Next day should reset.
    t1 = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    b3 = Bar(symbol="AAPL", time=t1, open=20, high=20, low=20, close=20, volume=10)
    engine.on_bar(b3)
    st = engine.symbol_states["AAPL"]
    assert st.indicators["session_day"] == t1.date()
    assert st.indicators["cum_vol"] == pytest.approx(10.0)
    assert st.indicators["session_vwap"] == pytest.approx(20.0)
    assert b3.vwap == pytest.approx(20.0)
