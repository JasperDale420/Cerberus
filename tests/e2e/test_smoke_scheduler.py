from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.scheduler import CerberusScheduler


def _write_yaml(path: Path, obj: dict) -> None:
    path.write_text(yaml.safe_dump(obj))


@pytest.mark.e2e
def test_cerberus_scheduler_smoke(tmp_path: Path) -> None:
    """
    Smoke test for the CerberusScheduler.
    Verifies it starts job registration and can trigger the subprocess logic.
    """
    # 1. Setup Config
    config_path = tmp_path / "config.yaml"
    config_dict = {
        "timezone": "UTC",
        "schedule_time": "09:30",
        "config_path": str(config_path),  # Pass explicit config path
    }
    _write_yaml(config_path, config_dict)

    # 2. Initialize Service
    scheduler = CerberusScheduler(config_dict)

    # 3. Verify Scheduler Setup
    # We can't easily start the BlockingScheduler in a test without blocking.
    # But we can verify it added the job.
    # Note: start() adds the job.

    # Let's mock the internal scheduler.add_job to verify registration logic
    with patch.object(scheduler.scheduler, "add_job") as mock_add_job:
        with patch.object(scheduler.scheduler, "start"):  # Prevent blocking
            scheduler.start()
            mock_add_job.assert_called_once()
            args, kwargs = mock_add_job.call_args
            assert kwargs["id"] == "cerberus_daily_session"
            assert kwargs["name"] == "Cerberus Daily Trading Session"

    # 4. Verify Subprocess Launch Logic
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Manually trigger the private method to verify command construction
        scheduler._run_daily_session()

        mock_subprocess.assert_called_once()
        args, _ = mock_subprocess.call_args
        cmd_list = args[0]

        # Verify command structure: [exec, -m, src.main, --mode, live, --config, path]
        assert "src.main" in cmd_list
        # We ensured config["config_path"] was set
        assert "--config" in cmd_list
        assert str(config_path) in cmd_list
