from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.agent.core import Agent
from src.agent.models import ActionType, AgentAction
from src.core.config import ConfigLoader


@pytest.mark.unit
def test_agent_writes_strategies_auto_next_to_active_config_suite(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({}))
    (tmp_path / "strategies.yaml").write_text(yaml.safe_dump({"strategies": {"vwap_reversion": {"enabled": True}}}))

    logger = MagicMock()
    loader = ConfigLoader(config_dir=str(tmp_path))

    agent = Agent(
        logger,
        loader,
        # Leave config_path as default; pass config suite as a file path.
        config_path_or_dir=str(tmp_path / "config.yaml"),
    )

    assert Path(agent.config_path) == (tmp_path / "strategies.auto.yaml")

    ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    agent.apply_actions(
        [
            AgentAction(
                timestamp=ts,
                action_type=ActionType.TUNE_PARAM,
                strategy="vwap_reversion",
                regime="CHOP",
                details={"new_params": {"band_sigma": 1.5}, "window_days": 30},
                reason="unit test",
            )
        ]
    )

    assert (tmp_path / "strategies.auto.yaml").exists()
