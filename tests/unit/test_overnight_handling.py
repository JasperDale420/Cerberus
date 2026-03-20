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
