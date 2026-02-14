from datetime import datetime

import pytest

from src.core.domain import Bar, MarketState, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class CaptureLogger(StructuredLogger):
    def __init__(self):
        self.errors = []
        self.warnings = []

    def info(self, msg, **kwargs):
        pass

    def warning(self, msg, **kwargs):
        self.warnings.append((msg, kwargs))

    def error(self, msg, **kwargs):
        self.errors.append((msg, kwargs))


class DummyStrategy(BaseStrategy):
    name = "dummy"

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        return None


@pytest.mark.unit
def test_hard_stop_invalid_format_raises_and_logs():
    logger = CaptureLogger()
    strat = DummyStrategy({"hard_stop_time": "25:99"}, logger)

    with pytest.raises(ValueError, match="Invalid hard_stop_time"):
        strat.is_past_hard_stop(datetime(2023, 1, 1, 10, 0))

    assert logger.errors
    msg, payload = logger.errors[0]
    assert msg == "Invalid hard stop time"
    assert payload.get("hard_stop_time") == "25:99"


@pytest.mark.unit
def test_hard_stop_valid_time_enforces_cutoff():
    logger = CaptureLogger()
    strat = DummyStrategy({"hard_stop_time": "11:00"}, logger)

    assert strat.is_past_hard_stop(datetime(2023, 1, 1, 10, 59)) is False
    assert strat.is_past_hard_stop(datetime(2023, 1, 1, 11, 0)) is True
