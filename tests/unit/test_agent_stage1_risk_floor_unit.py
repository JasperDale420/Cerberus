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
def test_stage1_reduce_risk_can_reach_zero_floor(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"agent": {"stage1": {}}}))
    (tmp_path / "risk.yaml").write_text(
        yaml.safe_dump({"risk": {"max_risk_per_trade": 1.0}})
    )
    (tmp_path / "strategies.yaml").write_text(
        yaml.safe_dump({"strategies": {"vwap_reversion": {"enabled": True}}})
    )

    logger = MagicMock()
    loader = ConfigLoader(config_dir=str(tmp_path))
    agent = Agent(
        logger,
        loader,
        config_path=str(tmp_path / "strategies.auto.yaml"),
        config_path_or_dir=str(tmp_path),
    )

    ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    actions = [
        AgentAction(
            timestamp=ts,
            action_type=ActionType.REDUCE_RISK,
            strategy="vwap_reversion",
            regime=None,
            details={},
            reason="unit",
        )
        for _ in range(10)
    ]

    agent.apply_actions(actions)

    raw = yaml.safe_load((tmp_path / "strategies.auto.yaml").read_text())
    assert raw["vwap_reversion"]["max_risk_per_trade"] == 0.0


@pytest.mark.unit
def test_stage1_reduce_risk_can_reach_zero_floor_per_regime(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"agent": {"stage1": {}}}))
    (tmp_path / "risk.yaml").write_text(
        yaml.safe_dump({"risk": {"max_risk_per_trade": 1.0}})
    )
    (tmp_path / "strategies.yaml").write_text(
        yaml.safe_dump({"strategies": {"vwap_reversion": {"enabled": True}}})
    )

    logger = MagicMock()
    loader = ConfigLoader(config_dir=str(tmp_path))
    agent = Agent(
        logger,
        loader,
        config_path=str(tmp_path / "strategies.auto.yaml"),
        config_path_or_dir=str(tmp_path),
    )

    ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    actions = [
        AgentAction(
            timestamp=ts,
            action_type=ActionType.REDUCE_RISK,
            strategy="vwap_reversion",
            regime="chop",
            details={},
            reason="unit",
        )
        for _ in range(10)
    ]

    agent.apply_actions(actions)

    raw = yaml.safe_load((tmp_path / "strategies.auto.yaml").read_text())
    assert raw["vwap_reversion"]["regimes"]["chop"]["max_risk_per_trade"] == 0.0
