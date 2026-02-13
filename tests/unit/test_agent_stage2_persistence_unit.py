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
def test_stage2_tune_param_persists_and_is_flattened_into_runtime_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "stage1": {
                        "min_trades": 20,
                        "z_high": 1.645,
                        "max_drawdown_r": 10.0,
                    },
                    "stage2": {"enabled": True, "window_days": 30, "search_space": {}},
                }
            }
        )
    )
    (tmp_path / "strategies.yaml").write_text(
        yaml.safe_dump({"strategies": {"vwap_reversion": {"enabled": True, "band_sigma": 2.0}}})
    )
    (tmp_path / "risk.yaml").write_text(yaml.safe_dump({"risk": {"max_risk_per_trade": 5.0}}))
    (tmp_path / "scanner.yaml").write_text(yaml.safe_dump({"scanner": {"max_watchlist_size": 10}}))
    (tmp_path / "universe.yaml").write_text(yaml.safe_dump({"universe": {"symbols": ["SPY"]}}))
    (tmp_path / "logging.yaml").write_text(yaml.safe_dump({"logging": {"format": "json"}}))

    logger = MagicMock()
    loader = ConfigLoader(config_dir=str(tmp_path))

    agent = Agent(
        logger,
        loader,
        config_path=str(tmp_path / "strategies.auto.yaml"),
        config_path_or_dir=str(tmp_path),
    )

    ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    action = AgentAction(
        timestamp=ts,
        action_type=ActionType.TUNE_PARAM,
        strategy="vwap_reversion",
        regime="CHOP",
        details={
            "new_params": {"band_sigma": 2.5},
            "metrics": {"expectancy": 0.2, "max_drawdown_r": 1.0, "n_trades": 50},
            "window_days": 30,
        },
        reason="unit test",
    )

    agent.apply_actions([action])

    raw = yaml.safe_load((tmp_path / "strategies.auto.yaml").read_text())
    assert raw["vwap_reversion"]["params"]["band_sigma"] == 2.5
    assert raw["vwap_reversion"]["metadata"]["last_optimized"] == "2025-01-02"
    assert raw["vwap_reversion"]["metadata"]["window_days"] == 30

    cfg = loader.load_config(str(tmp_path))
    assert cfg["strategies"]["vwap_reversion"]["band_sigma"] == 2.5
    assert cfg["strategies"]["vwap_reversion"]["params"]["band_sigma"] == 2.5
