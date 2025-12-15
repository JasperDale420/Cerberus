import os
from datetime import date
from unittest.mock import MagicMock

import pytest

from src.agent.core import ActionType, Agent
from src.agent.models import StrategyDailyStats


@pytest.fixture
def mock_stats():
    return StrategyDailyStats(
        date=date(2023, 1, 1),
        strategy="vwap_reversion",
        regime="CHOP",
        n_trades=10,
        winrate=0.4,
        avg_r=1.0,
        std_r=0.5,
        max_drawdown_r=12.0,
        expectancy=-0.6,
        total_pnl_r=-6.0,
    )


def test_tune_parameters(mock_stats):
    logger = MagicMock()
    config_loader = MagicMock()
    config_loader.get_env.return_value = "http://localhost:8000"

    agent = Agent(logger, config_loader)

    # Mock LLM response
    mock_response = '{"band_sigma": 2.5}'
    agent.llm_client = MagicMock()
    agent.llm_client.complete.return_value = mock_response

    current_config = {"band_sigma": 2.0}

    actions = agent.tune_parameters(mock_stats, current_config)

    assert len(actions) == 1
    action = actions[0]
    assert action.action_type == ActionType.TUNE_PARAM
    assert action.details["new_params"] == {"band_sigma": 2.5}

    # Verify LLM call
    agent.llm_client.complete.assert_called_once()


def test_propose_code_changes(mock_stats, tmp_path):
    logger = MagicMock()
    config_loader = MagicMock()
    config_loader.get_env.return_value = "http://localhost:8000"

    agent = Agent(logger, config_loader)

    # Create dummy strategy file
    strat_file = tmp_path / "vwap_reversion.py"
    strat_file.write_text("class VWAPReversionStrategy: pass")

    # Mock LLM response
    mock_code = "class VWAPReversionStrategyV2: pass"
    agent.llm_client = MagicMock()
    agent.llm_client.complete.return_value = mock_code

    # Run
    # We need to patch os.makedirs and open inside the method, OR just let it write to a temp dir if we can control it.
    # The method writes to "src/strategies/proposals". We should mock this path or run in a temp cwd.
    # Easier to mock open/makedirs or just let it write if we clean up.
    # Let's mock the file writing part to avoid cluttering the actual project.

    # Create the directory
    os.makedirs("src/strategies/proposals", exist_ok=True)

    actions = agent.propose_code_changes(mock_stats, str(strat_file))

    assert len(actions) == 1
    action = actions[0]
    assert action.action_type == ActionType.CODE_PROPOSAL
    assert "proposal_path" in action.details

    # Check file content
    proposal_path = action.details["proposal_path"]
    assert os.path.exists(proposal_path)
    with open(proposal_path, "r") as f:
        content = f.read()
    assert content == mock_code

    # Cleanup
    os.remove(proposal_path)
