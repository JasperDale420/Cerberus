from unittest.mock import MagicMock

import pytest

from src.core.domain import (
    LiquidityRegime,
    MarketRegimeSnapshot,
    MarketState,
    Regime,
    RiskRegime,
    SessionRegime,
    TrendRegime,
    VolRegime,
)
from src.engine.strategy_engine import (
    StrategyActivationPolicy,
    StrategyEngine,
    StrategyRouting,
)


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_strategy():
    s = MagicMock()
    s.on_bar.return_value = None
    return s


@pytest.fixture
def engine(mock_logger, mock_strategy):
    routing = StrategyRouting(strategies_by_regime={Regime.CHOP: ["strat1"], Regime.BULL: ["strat2"]})
    strategies = {"strat1": mock_strategy, "strat2": mock_strategy}
    return StrategyEngine(strategies, routing, mock_logger)


@pytest.mark.unit
def test_get_active_strategies(engine):
    symbol_state = MagicMock()
    symbol_state.allowed_strategies = ["strat1", "strat2", "strat3"]

    market_state = MarketState(time=MagicMock(), regime=Regime.CHOP)

    active = engine._get_active_strategies(symbol_state, market_state)
    assert active == ["strat1"]

    market_state.regime = Regime.BULL
    active = engine._get_active_strategies(symbol_state, market_state)
    assert active == ["strat2"]  # Sorted.


@pytest.mark.unit
def test_on_bar_runs_active_strategies(engine, mock_strategy):
    symbol_state = MagicMock()
    symbol_state.allowed_strategies = ["strat1"]
    market_state = MarketState(time=MagicMock(), regime=Regime.CHOP)
    bar = MagicMock()

    mock_strategy.on_bar.return_value = "SIGNAL"

    signals = engine.on_bar("AAPL", bar, symbol_state, market_state)
    assert signals == ["SIGNAL"]
    mock_strategy.on_bar.assert_called_once()


@pytest.mark.unit
def test_safe_run_strategy_missing(engine, mock_logger):
    # Strategy name 'missing' not in map
    res = engine._safe_run_strategy("missing", "AAPL", MagicMock(), MagicMock(), MagicMock())
    assert res is None
    mock_logger.error.assert_called_with(
        "Strategy missing from registry — signals will not be generated", strategy="missing"
    )


@pytest.mark.unit
def test_safe_run_strategy_exception(engine, mock_strategy, mock_logger):
    mock_strategy.on_bar.side_effect = RuntimeError("Boom")

    # Callback
    on_error = MagicMock()
    engine.on_error = on_error

    bar = MagicMock()
    symbol_state = MagicMock()
    market_state = MarketState(time=MagicMock(), regime=Regime.CHOP)

    res = engine._safe_run_strategy("strat1", "AAPL", bar, symbol_state, market_state)

    assert res is None
    on_error.assert_called_once()
    mock_logger.error.assert_called()


@pytest.mark.unit
def test_safe_run_strategy_exception_callback_error(engine, mock_strategy, mock_logger):
    mock_strategy.on_bar.side_effect = RuntimeError("Boom")
    on_error = MagicMock(side_effect=RuntimeError("Callback Fail"))
    engine.on_error = on_error

    # Should not crash
    res = engine._safe_run_strategy(
        "strat1",
        "AAPL",
        MagicMock(),
        MagicMock(),
        MarketState(time=MagicMock(), regime=Regime.CHOP),
    )

    assert res is None
    mock_logger.error.assert_called()


@pytest.mark.unit
def test_on_bar_multiple_signals(engine, mock_strategy):
    # Setup strat1 and strat2 both allowed and both in regime (hack routing)
    engine.routing = StrategyRouting(strategies_by_regime={Regime.CHOP: ["strat1", "strat2"]})

    symbol_state = MagicMock()
    symbol_state.allowed_strategies = ["strat1", "strat2"]
    market_state = MarketState(time=MagicMock(), regime=Regime.CHOP)

    mock_strategy.on_bar.return_value = "SIG"

    signals = engine.on_bar("AAPL", MagicMock(), symbol_state, market_state)
    assert len(signals) == 2
    assert signals == ["SIG", "SIG"]


@pytest.mark.unit
def test_activation_policy_allows_strategy_when_snapshot_matches(mock_logger, mock_strategy):
    policy = StrategyActivationPolicy(
        session=(SessionRegime.OPENING,),
        trend=(TrendRegime.UP,),
        vol=(VolRegime.NORMAL,),
        liquidity=(LiquidityRegime.GOOD,),
        risk=(RiskRegime.NEUTRAL,),
        min_confidence=0.0,
    )
    routing = StrategyRouting(
        strategies_by_regime={Regime.CHOP: []},
        activation_policies={"strat1": policy},
    )
    engine = StrategyEngine({"strat1": mock_strategy}, routing, mock_logger)

    symbol_state = MagicMock()
    symbol_state.allowed_strategies = ["strat1"]

    snap = MarketRegimeSnapshot(
        time=MagicMock(),
        index_symbol="SPY",
        vol_symbol=None,
        trend=TrendRegime.UP,
        vol=VolRegime.NORMAL,
        liquidity=LiquidityRegime.GOOD,
        risk=RiskRegime.NEUTRAL,
        session=SessionRegime.OPENING,
        trend_strength=2.0,
        confidence={
            "trend": 1.0,
            "vol": 1.0,
            "liquidity": 1.0,
            "risk": 1.0,
            "session": 1.0,
        },
    )
    market_state = MarketState(time=MagicMock(), regime=Regime.CHOP, regime_snapshot=snap)

    assert engine._get_active_strategies(symbol_state, market_state) == ["strat1"]


@pytest.mark.unit
def test_activation_policy_falls_back_to_legacy_when_no_snapshot(mock_logger, mock_strategy):
    policy = StrategyActivationPolicy(trend=(TrendRegime.UP,))
    routing = StrategyRouting(
        strategies_by_regime={Regime.BULL: ["strat1"]},
        activation_policies={"strat1": policy},
    )
    engine = StrategyEngine({"strat1": mock_strategy}, routing, mock_logger)

    symbol_state = MagicMock()
    symbol_state.allowed_strategies = ["strat1"]
    market_state = MarketState(time=MagicMock(), regime=Regime.BULL, regime_snapshot=None)

    assert engine._get_active_strategies(symbol_state, market_state) == ["strat1"]
