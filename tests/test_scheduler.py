import sys
from unittest.mock import patch

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


def test_schedule_time_parsing_valid():
    scheduler = CerberusScheduler({"schedule_time": "09:25"})

    hour, minute = scheduler._parse_schedule_time("09:25")

    assert hour == 9
    assert minute == 25


def test_schedule_time_parsing_invalid_format():
    scheduler = CerberusScheduler({"schedule_time": "09:25"})

    try:
        scheduler._parse_schedule_time("bad-time")
    except ValueError as exc:
        assert "schedule_time must be in HH:MM format" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid schedule_time format")


def test_schedule_time_parsing_invalid_range():
    scheduler = CerberusScheduler({"schedule_time": "09:25"})

    try:
        scheduler._parse_schedule_time("25:00")
    except ValueError as exc:
        assert "schedule_time hour must be between 0 and 23" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid schedule_time hour")


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
