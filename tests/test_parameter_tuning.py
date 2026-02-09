from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.agent.models import StrategyDailyStats
from src.agent.stage2 import DeterministicStage2Evaluator, Stage2Metrics, Stage2Tuner


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_config_loader():
    loader = MagicMock()
    loader.load_config.return_value = {
        "agent": {
            "stage2": {
                "enabled": True,
                "window_days": 30,
                "search_space": {"vwap_reversion": {"band_sigma": [1.5, 2.5]}},
            }
        }
    }
    return loader


@pytest.fixture
def mock_evaluator():
    evaluator = MagicMock(spec=DeterministicStage2Evaluator)

    # Return different metrics for different params to test optimization
    def side_effect(strategy, regime, params, **kwargs):
        if params.get("band_sigma") == 2.5:
            return Stage2Metrics(expectancy=1.0, max_drawdown_r=2.0, n_trades=20)
        return Stage2Metrics(expectancy=0.5, max_drawdown_r=2.0, n_trades=20)

    evaluator.evaluate.side_effect = side_effect
    return evaluator


def test_tune_parameters_grid_search(mock_logger, mock_config_loader, mock_evaluator):
    tuner = Stage2Tuner(mock_logger, mock_config_loader, evaluator=mock_evaluator)

    stats = StrategyDailyStats(
        date=datetime.now(timezone.utc).date(),
        strategy="vwap_reversion",
        regime="bull",
        n_trades=20,
        winrate=0.5,
        avg_r=0.8,
        std_r=1.0,
        max_drawdown_r=5.0,
        expectancy=0.4,
        total_pnl_r=16.0,
    )

    current_config = {"band_sigma": 2.0}

    actions = tuner.tune_parameters(stats, current_config, min_trades=5, max_dd_r=10.0)

    assert len(actions) == 1
    assert actions[0].details["new_params"] == {"band_sigma": 2.5}
    assert actions[0].details["metrics"]["expectancy"] == 1.0


def test_tune_parameters_walk_forward(mock_logger, mock_config_loader, mock_evaluator):
    # Enable walk-forward in config
    mock_config_loader.load_config.return_value["agent"]["stage2"]["walk_forward"] = {
        "enabled": True,
        "n_windows": 2,
        "step_days": 7,
        "stability_threshold": 0.7,
    }

    tuner = Stage2Tuner(mock_logger, mock_config_loader, evaluator=mock_evaluator)

    stats = StrategyDailyStats(
        date=datetime.now(timezone.utc).date(),
        strategy="vwap_reversion",
        regime="bull",
        n_trades=20,
        winrate=0.5,
        avg_r=0.8,
        std_r=1.0,
        max_drawdown_r=5.0,
        expectancy=0.4,
        total_pnl_r=16.0,
    )

    current_config = {"band_sigma": 2.0}

    # Mock evaluator for stability check
    # Stability check calls evaluator for each window
    actions = tuner.tune_parameters(stats, current_config, min_trades=5, max_dd_r=10.0)

    assert len(actions) == 1
    assert actions[0].details["walk_forward"] is True
