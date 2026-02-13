from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.agent.core import ActionType, Agent
from src.agent.models import StrategyDailyStats


@pytest.mark.unit
@patch("src.agent.core.LLMClient")
def test_analyze_performance_z_score(mock_llm_cls):
    # Setup
    logger = MagicMock()
    config_loader = MagicMock()
    config_loader.load_config.return_value = {
        "agent": {
            "stage1": {
                "window_days": 30,
                "min_trades": 20,
                "z_high": 1.645,
                "max_drawdown_r": 4.0,
            }
        }
    }
    # Mock LLMClient instance
    mock_llm_instance = MagicMock()
    mock_llm_cls.return_value = mock_llm_instance

    agent = Agent(logger, config_loader)

    # 1. High stats, should NOT disable
    good_stats = StrategyDailyStats(
        date=date.today(),
        strategy="winning_strat",
        regime="BULL",
        n_trades=100,
        winrate=0.6,
        avg_r=0.5,
        std_r=1.0,
        max_drawdown_r=2.0,
        expectancy=0.3,  # Positive
        total_pnl_r=50.0,
    )

    # 2. Insufficient data, should skip
    low_data_stats = StrategyDailyStats(
        date=date.today(),
        strategy="new_strat",
        regime="CHOP",
        n_trades=5,  # < 20 min
        winrate=0.4,
        avg_r=-0.1,
        std_r=0.5,
        max_drawdown_r=0.5,
        expectancy=-0.1,
        total_pnl_r=-0.5,
    )

    # 3. Bad stats, Significant Negative Z-Score
    # Exp = -0.2, Std=0.5, N=100
    # SE = 0.5 / 10 = 0.05
    # Z = -0.2 / 0.05 = -4.0  (< -1.645) -> DISABLE
    bad_stats = StrategyDailyStats(
        date=date.today(),
        strategy="losing_strat",
        regime="BEAR",
        n_trades=100,
        winrate=0.3,
        avg_r=-0.6,
        std_r=0.5,
        max_drawdown_r=5.0,  # >= configured max_drawdown_r threshold
        expectancy=-0.2,
        total_pnl_r=-20.0,
    )

    # 4. Bad stats but NOT Significant
    # Exp = -0.05, Std=1.0, N=25
    # SE = 1.0 / 5 = 0.2
    # Z = -0.05 / 0.2 = -0.25 (> -1.645) -> NO ACTION
    noisy_stats = StrategyDailyStats(
        date=date.today(),
        strategy="noisy_strat",
        regime="CHOP",
        n_trades=25,
        winrate=0.45,
        avg_r=-0.1,
        std_r=1.0,
        max_drawdown_r=2.0,
        expectancy=-0.05,
        total_pnl_r=-1.25,
    )

    actions = agent.analyze_performance([good_stats, low_data_stats, bad_stats, noisy_stats])

    # Assertions
    assert len(actions) == 1
    action = actions[0]
    assert action.strategy == "losing_strat"
    assert action.action_type == ActionType.DISABLE_STRATEGY
    assert "z_score" in action.details
    assert action.details["z_score"] < -1.645
