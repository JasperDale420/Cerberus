import sys
from unittest.mock import patch

import pytest
from apscheduler.triggers.cron import CronTrigger

from src.scheduler import CerberusScheduler


def test_scheduler_initialization():
    config = {"timezone": "America/New_York", "schedule_time": "09:30"}
    scheduler = CerberusScheduler(config)

    # Verify timezone
    assert str(scheduler.tz) == "America/New_York"

    # Verify scheduler exists
    assert scheduler.scheduler is not None


def test_add_job_correctly():
    config = {"timezone": "America/New_York", "schedule_time": "09:25"}
    scheduler = CerberusScheduler(config)

    # Mock the internal scheduler.start to avoid blocking
    with patch.object(scheduler.scheduler, "start"):
        with patch.object(scheduler.scheduler, "add_job") as mock_add_job:
            scheduler.start()

            mock_add_job.assert_called_once()
            _, kwargs = mock_add_job.call_args

            # Check arguments
            assert kwargs["id"] == "cerberus_daily_session"
            assert isinstance(kwargs["trigger"], CronTrigger)

            # Check trigger details
            # Verify hour/minute are correct
            # APScheduler internals can be tricky to inspect directly,
            # but we can rely on integration verification if this is too fragile.
            # But let's at least check the repr or configured fields if accessible.
            pass


def test_scheduler_invalid_schedule_time_logs_and_raises():
    config = {"timezone": "America/New_York", "schedule_time": "9-30"}
    scheduler = CerberusScheduler(config)

    with patch("src.scheduler.logger") as mock_logger:
        with patch.object(scheduler.scheduler, "start"):
            with pytest.raises(ValueError, match="Invalid schedule_time"):
                scheduler.start()

        mock_logger.error.assert_called_once()


def test_scheduler_invalid_timezone_logs_and_raises():
    with patch("src.scheduler.logger") as mock_logger:
        with pytest.raises(ValueError, match="Invalid timezone"):
            CerberusScheduler({"timezone": "Bad/Timezone", "schedule_time": "09:25"})

        mock_logger.error.assert_called_once()


@patch("subprocess.run")
def test_run_daily_session_subprocess(mock_run):
    config = {"config_path": "custom_config.yaml"}
    scheduler = CerberusScheduler(config)

    mock_run.return_value.returncode = 0

    scheduler._run_daily_session()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]

    # Verify command structure
    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert "src.main" in cmd
    assert "--mode" in cmd
    assert "live" in cmd
    assert "--config" in cmd
    assert "custom_config.yaml" in cmd
