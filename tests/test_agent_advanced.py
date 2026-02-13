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
    config_loader.load_config.return_value = {
        "agent": {
            "stage2": {
                "enabled": True,
                "search_space": {"vwap_reversion": {"band_sigma": [2.0, 2.5]}},
            }
        }
    }

    def eval_fn(_stats, cfg):
        # Prefer higher band_sigma deterministically.
        band = float(cfg.get("band_sigma", 0.0))
        return {"expectancy": band, "max_drawdown_r": 0.0, "n_trades": 100}

    agent = Agent(logger, config_loader, stage2_evaluator=eval_fn)

    current_config = {"band_sigma": 2.0}

    actions = agent.tune_parameters(mock_stats, current_config)

    assert len(actions) == 1
    action = actions[0]
    assert action.action_type == ActionType.TUNE_PARAM
    assert action.details["new_params"] == {"band_sigma": 2.5}


def test_propose_code_changes(mock_stats, tmp_path):
    logger = MagicMock()
    config_loader = MagicMock()
    config_loader.get_env.return_value = "http://localhost:8000"
    config_loader.load_config.return_value = {
        "strategies": {
            "strategies": {
                "vwap_reversion": {
                    "enabled": True,
                    "band_sigma": 2.0,
                }
            }
        },
        "agent": {
            "stage3": {
                "enabled": True,
                "approval_env_var": "CERBERUS_STAGE3_APPROVED",
                "write_to_src": True,
                "propose_scanner_profiles": False,
                "backtest": {
                    "enabled": True,
                    "min_trades": 0,
                    "max_drawdown_r": 999.0,
                    "min_expectancy_delta": 0.0,
                },
            }
        },
    }

    def stage3_eval(_stats, _source, _candidate, _cfg):
        return {
            "baseline_metrics": {
                "expectancy": 0.0,
                "max_drawdown_r": 0.0,
                "n_trades": 0,
            },
            "candidate_metrics": {
                "expectancy": 1.0,
                "max_drawdown_r": 0.0,
                "n_trades": 1,
            },
        }

    # Create dummy strategy file
    strat_file = tmp_path / "vwap_reversion.py"
    strat_file.write_text("class VWAPReversionStrategy: pass")

    # Mock LLM response
    mock_code = "class VWAPReversionStrategyV2: pass"
    mock_llm = MagicMock()
    mock_llm.complete.return_value = mock_code

    agent = Agent(logger, config_loader, stage3_evaluator=stage3_eval, llm_client=mock_llm)

    # Run
    # We need to patch os.makedirs and open inside the method, OR just let it write to a temp dir if we can control it.
    # The method writes to "src/strategies/proposals". We should mock this path or run in a temp cwd.
    # Easier to mock open/makedirs or just let it write if we clean up.
    # Let's mock the file writing part to avoid cluttering the actual project.

    os.environ["CERBERUS_STAGE3_APPROVED"] = "true"
    os.makedirs("src/strategies/proposals", exist_ok=True)

    actions = agent.propose_code_changes(mock_stats, str(strat_file))

    assert len(actions) == 1
    action = actions[0]
    assert action.action_type == ActionType.CODE_PROPOSAL
    assert "proposal_file" in action.details
    assert "diff_file" in action.details
    assert "summary_file" in action.details

    # Check file content
    proposal_path = action.details["proposal_file"]
    assert os.path.exists(proposal_path)
    with open(proposal_path, "r") as f:
        content = f.read()
    assert content == mock_code
    assert os.path.exists(action.details["diff_file"])
    assert os.path.exists(action.details["summary_file"])

    # Cleanup
    os.remove(proposal_path)
    os.remove(action.details["diff_file"])
    os.remove(action.details["summary_file"])
    os.environ.pop("CERBERUS_STAGE3_APPROVED", None)


def test_propose_code_changes_can_emit_scanner_profile_proposal(mock_stats, tmp_path):
    logger = MagicMock()
    config_loader = MagicMock()
    config_loader.get_env.return_value = "http://localhost:8000"
    config_loader.load_config.return_value = {
        "strategies": {
            "strategies": {
                "vwap_reversion": {
                    "enabled": True,
                    "band_sigma": 2.0,
                }
            }
        },
        "agent": {
            "stage3": {
                "enabled": True,
                "approval_env_var": "CERBERUS_STAGE3_APPROVED",
                "write_to_src": True,
                "propose_scanner_profiles": True,
                "backtest": {
                    "enabled": True,
                    "min_trades": 0,
                    "max_drawdown_r": 999.0,
                    "min_expectancy_delta": 0.0,
                },
            }
        },
    }

    def stage3_eval(_stats, _source, _candidate, _cfg):
        return {
            "baseline_metrics": {
                "expectancy": 0.0,
                "max_drawdown_r": 0.0,
                "n_trades": 0,
            },
            "candidate_metrics": {
                "expectancy": 1.0,
                "max_drawdown_r": 0.0,
                "n_trades": 1,
            },
        }

    # Mock LLM response
    mock_llm = MagicMock()
    mock_llm.complete.return_value = (
        '{"strategy_code":"class VWAPReversionStrategyV2: pass\\n","scanner_profiles_code":"# scanner profiles v2\\n"}'
    )

    agent = Agent(logger, config_loader, stage3_evaluator=stage3_eval, llm_client=mock_llm)

    strat_file = tmp_path / "vwap_reversion.py"
    strat_file.write_text("class VWAPReversionStrategy: pass")

    os.environ["CERBERUS_STAGE3_APPROVED"] = "true"
    os.makedirs("src/strategies/proposals", exist_ok=True)

    actions = agent.propose_code_changes(mock_stats, str(strat_file))
    assert len(actions) == 1
    action = actions[0]
    assert action.action_type == ActionType.CODE_PROPOSAL

    assert action.details["scanner_profiles_proposal_file"] is not None
    assert action.details["scanner_profiles_diff_file"] is not None
    assert os.path.exists(action.details["scanner_profiles_proposal_file"])
    assert os.path.exists(action.details["scanner_profiles_diff_file"])

    os.remove(action.details["proposal_file"])
    os.remove(action.details["diff_file"])
    os.remove(action.details["summary_file"])
    os.remove(action.details["scanner_profiles_proposal_file"])
    os.remove(action.details["scanner_profiles_diff_file"])
    os.environ.pop("CERBERUS_STAGE3_APPROVED", None)
