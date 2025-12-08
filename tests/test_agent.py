import pytest
import os
import yaml
from datetime import date, datetime
from unittest.mock import MagicMock
from src.agent.core import Agent, ActionType
from src.agent.models import StrategyDailyStats
from src.core.config import ConfigLoader

@pytest.fixture
def mock_stats():
    return StrategyDailyStats(
        date=date.today(),
        strategy="vwap_reversion",
        regime="CHOP",
        n_trades=10,
        winrate=0.4,
        avg_r=1.0,
        std_r=0.5,
        max_drawdown_r=12.0, # > 10.0 threshold
        expectancy=-0.6, # < -0.5 threshold
        total_pnl_r=-6.0
    )

def test_analyze_performance(mock_stats):
    logger = MagicMock()
    agent = Agent(logger)
    
    actions = agent.analyze_performance([mock_stats])
    
    assert len(actions) == 1 # Should stop after disable?
    # Logic: if expectancy < -0.5, it appends DISABLE and continues (skips other checks for THAT strategy)
    
    action = actions[0]
    assert action.action_type == ActionType.DISABLE_STRATEGY
    assert action.strategy == "vwap_reversion"
    assert action.reason == "Expectancy below threshold (-0.5 R)"

def test_apply_actions(tmp_path):
    logger = MagicMock()
    config_path = tmp_path / "strategies.auto.yaml"
    agent = Agent(logger, config_path=str(config_path))
    
    # Create action
    action = MagicMock()
    action.strategy = "vwap_reversion"
    action.action_type = ActionType.DISABLE_STRATEGY
    action.applied = False
    
    # Apply
    agent.apply_actions([action])
    
    # Verify file created
    assert config_path.exists()
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    assert data["vwap_reversion"]["enabled"] is False
    assert action.applied is True

def test_config_loader_merge(tmp_path):
    # Setup main config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    main_config_path = config_dir / "config.yaml"
    auto_config_path = config_dir / "strategies.auto.yaml"
    
    with open(main_config_path, "w") as f:
        yaml.dump({"strategies": {"vwap_reversion": {"enabled": True, "risk": 1.0}}}, f)
        
    with open(auto_config_path, "w") as f:
        yaml.dump({"vwap_reversion": {"enabled": False}}, f)
        
    # Load
    loader = ConfigLoader()
    config = loader.load_config(str(main_config_path))
    
    # Verify merge
    assert config["strategies"]["vwap_reversion"]["enabled"] is False
    assert config["strategies"]["vwap_reversion"]["risk"] == 1.0 # Preserved
