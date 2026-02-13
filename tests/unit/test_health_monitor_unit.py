from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.engine.health import HealthMonitor


@pytest.mark.unit
def test_record_error_recovers_from_non_int_count() -> None:
    logger = MagicMock()
    monitor = HealthMonitor(config={}, logger=logger, clock=lambda: datetime.now(timezone.utc))
    monitor.error_counts["execution"] = "bad"

    monitor.record_error("execution")

    assert monitor.error_counts["execution"] == 1
    logger.warning.assert_called_once()


@pytest.mark.unit
def test_record_error_unknown_module_initializes_count() -> None:
    logger = MagicMock()
    monitor = HealthMonitor(config={}, logger=logger, clock=lambda: datetime.now(timezone.utc))

    monitor.record_error("new_module")

    assert monitor.error_counts["new_module"] == 1
    logger.warning.assert_not_called()
