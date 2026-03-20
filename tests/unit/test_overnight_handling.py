from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_base_strategy_overnight_defaults():
    from src.strategies.base import BaseStrategy

    class StubStrategy(BaseStrategy):
        name = "stub"

        def on_bar(self, symbol, bar, symbol_state, market_state):
            return None

    strat = StubStrategy(config={}, logger=MagicMock())
    assert strat.allow_overnight is False
    assert strat.max_hold_days == 0
    assert strat.overnight_stop_mult == 1.0


@pytest.mark.unit
def test_base_strategy_overnight_from_config():
    from src.strategies.base import BaseStrategy

    class StubStrategy(BaseStrategy):
        name = "stub"

        def on_bar(self, symbol, bar, symbol_state, market_state):
            return None

    cfg = {"allow_overnight": True, "max_hold_days": 5, "overnight_stop_mult": 1.5}
    strat = StubStrategy(config=cfg, logger=MagicMock())
    assert strat.allow_overnight is True
    assert strat.max_hold_days == 5
    assert strat.overnight_stop_mult == 1.5


@pytest.mark.unit
def test_should_flatten_position_intraday_strategy():
    from src.backtest.runner import _should_flatten_position

    class FakeStrategy:
        allow_overnight = False
        max_hold_days = 0

    assert _should_flatten_position(FakeStrategy(), hold_days=0) is True


@pytest.mark.unit
def test_should_not_flatten_overnight_strategy():
    from src.backtest.runner import _should_flatten_position

    class FakeStrategy:
        allow_overnight = True
        max_hold_days = 0

    assert _should_flatten_position(FakeStrategy(), hold_days=0) is False


@pytest.mark.unit
def test_should_flatten_overnight_max_hold_exceeded():
    from src.backtest.runner import _should_flatten_position

    class FakeStrategy:
        allow_overnight = True
        max_hold_days = 3

    assert _should_flatten_position(FakeStrategy(), hold_days=2) is False
    assert _should_flatten_position(FakeStrategy(), hold_days=3) is True
    assert _should_flatten_position(FakeStrategy(), hold_days=5) is True


@pytest.mark.unit
def test_position_strategy_name_field():
    """Verify Position has strategy_name field and it defaults to empty string."""
    from src.core.domain import Position, Side

    pos = Position(
        symbol="AAPL",
        side=Side.LONG,
        qty=100,
        avg_price=150.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        strategy="mean_reversion",
    )
    assert pos.strategy_name == ""

    pos2 = Position(
        symbol="AAPL",
        side=Side.LONG,
        qty=100,
        avg_price=150.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        strategy="mean_reversion",
        strategy_name="mean_reversion",
    )
    assert pos2.strategy_name == "mean_reversion"
