from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Dict, Any, List
from enum import Enum

class ActionType(str, Enum):
    DISABLE_STRATEGY = "DISABLE_STRATEGY"
    ENABLE_STRATEGY = "ENABLE_STRATEGY"
    REDUCE_RISK = "REDUCE_RISK"
    INCREASE_RISK = "INCREASE_RISK"
    TUNE_PARAM = "TUNE_PARAM"
    CODE_PROPOSAL = "CODE_PROPOSAL"

@dataclass
class StrategyDailyStats:
    date: date
    strategy: str
    regime: str
    n_trades: int
    winrate: float
    avg_r: float
    std_r: float
    max_drawdown_r: float
    expectancy: float # avg_r * winrate - (1-winrate) * avg_loss_r (simplified, or just mean R)
    total_pnl_r: float

@dataclass
class AgentAction:
    timestamp: datetime
    action_type: ActionType
    strategy: str
    regime: Optional[str]
    details: Dict[str, Any]
    reason: str
    applied: bool = False
