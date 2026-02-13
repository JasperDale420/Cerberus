from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.logger import StructuredLogger
from src.engine.execution import ExecutionEngine


class _MockLogger(StructuredLogger):
    def __init__(self):
        self.logs = []

    def info(self, msg, **kwargs):
        self.logs.append(("INFO", msg, kwargs))

    def warning(self, msg, **kwargs):
        self.logs.append(("WARNING", msg, kwargs))

    def error(self, msg, **kwargs):
        self.logs.append(("ERROR", msg, kwargs))

    def bind(self, **kwargs):
        return self


class _MockConfig(dict):
    pass


@pytest.mark.unit
def test_on_bar_resets_error_count_on_success():
    logger = _MockLogger()
    engine = ExecutionEngine(
        config={"max_consecutive_errors": 5},
        logger=logger,
        clock=lambda: datetime(2023, 1, 1, tzinfo=timezone.utc),
    )
    engine.consecutive_on_bar_errors = 3

    # Simulate partial bad state but successful call
    # We pass a minimal bar object
    class Bar:
        symbol = "AAPL"
        time = None
        close = 100

    engine.on_bar(Bar())

    assert engine.consecutive_on_bar_errors == 0


@pytest.mark.unit
def test_on_bar_increments_error_count_on_failure():
    logger = _MockLogger()
    engine = ExecutionEngine(config={"max_consecutive_errors": 5, "index_symbol": "SPY"}, logger=logger)

    # Manipulate internal state to cause a crash inside on_bar
    # e.g. break symbol_states to cause KeyError or similar
    engine.symbol_states = None  # type: ignore

    # Should not raise yet
    class Bar:
        symbol = "AAPL"

    engine.on_bar(Bar())

    assert engine.consecutive_on_bar_errors == 1


@pytest.mark.unit
def test_on_bar_crashes_after_threshold():
    logger = _MockLogger()
    max_err = 3
    engine = ExecutionEngine(config={"max_consecutive_errors": max_err}, logger=logger)

    # Helper to force error
    engine.symbol_states = None  # type: ignore

    class Bar:
        symbol = "AAPL"

    # 1. Error
    engine.on_bar(Bar())
    assert engine.consecutive_on_bar_errors == 1

    # 2. Error
    engine.on_bar(Bar())
    assert engine.consecutive_on_bar_errors == 2

    # 3. Error -> Crash
    with pytest.raises(RuntimeError, match="Max consecutive execution errors exceeded"):
        engine.on_bar(Bar())

    # Verify we logged critical error before crashing
    assert any(
        log_entry[0] == "ERROR" and "Max consecutive execution errors exceeded" in log_entry[1]
        for log_entry in logger.logs
    )
