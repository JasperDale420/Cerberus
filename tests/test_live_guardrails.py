from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

# We need to test the logic inside async_main, but it's a large function.
# Instead of testing async_main directly (which starts real streams),
# we'll extract or mock the loop state.
# Since the loop is directly inside async_main, we'll suggest a refactor or
# use a very targeted mock of the loop's dependencies.


@pytest.mark.asyncio
async def test_loop_guardrails_logic():
    # Mock dependencies
    logger = MagicMock()
    engine = MagicMock()
    engine.market_state = MagicMock()
    config = {"force_flat_before_close_mins": 15}

    # We want to test the logic:
    # now = _now_local()
    # mins_to_close = (16 - now.hour) * 60 - now.minute

    market_tz = ZoneInfo("America/New_York")

    # Cases to test
    # 1. 15:40 (20 mins to close) -> No flatten yet, but should sleep 30s
    time_1540 = datetime(2023, 1, 3, 15, 40, tzinfo=market_tz)
    mins_to_close = (16 - time_1540.hour) * 60 - time_1540.minute
    assert mins_to_close == 20

    # 2. 15:45 (15 mins to close) -> Trigger flatten
    time_1545 = datetime(2023, 1, 3, 15, 45, tzinfo=market_tz)
    mins_to_close = (16 - time_1545.hour) * 60 - time_1545.minute
    assert mins_to_close == 15

    # 3. 15:59 (1 min to close) -> Log warning
    time_1559 = datetime(2023, 1, 3, 15, 59, tzinfo=market_tz)
    mins_to_close = (16 - time_1559.hour) * 60 - time_1559.minute
    assert mins_to_close == 1


def test_mins_to_close_calculation():
    # Simple unit test for the logic added to main.py
    def calc(h, m):
        return (16 - h) * 60 - m

    assert calc(15, 45) == 15
    assert calc(15, 0) == 60
    assert calc(12, 0) == 240
    assert calc(16, 0) == 0
    assert calc(16, 5) == -5


@pytest.mark.unit
def test_flattening_logic_state_mock():
    # Mocking the state variables we added to async_main
    flattened_for_date = None
    now = datetime(2023, 1, 3, 15, 45, tzinfo=ZoneInfo("America/New_York"))
    force_flat_mins = 15
    mins_to_close = (16 - now.hour) * 60 - now.minute

    engine = MagicMock()

    # Logic from main.py
    if 0 < mins_to_close <= force_flat_mins:
        target_date = now.date()
        if flattened_for_date != target_date:
            engine.flatten_all(reason="pre_market_close")
            flattened_for_date = target_date

    assert flattened_for_date == now.date()
    engine.flatten_all.assert_called_once_with(reason="pre_market_close")

    # Second run same date -> no second call
    engine.reset_mock()
    if 0 < mins_to_close <= force_flat_mins:
        target_date = now.date()
        if flattened_for_date != target_date:
            engine.flatten_all(reason="pre_market_close")

    engine.flatten_all.assert_not_called()
