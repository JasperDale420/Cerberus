from typing import Dict, Optional

from pydantic import BaseModel, Field


class RegimeConfig(BaseModel):
    enabled: bool = True
    max_risk_per_trade: Optional[float] = None
    stop_multiplier: Optional[float] = None
    take_profit_multiplier: Optional[float] = None


class StrategyConfig(BaseModel):
    enabled: bool = True
    max_risk_per_trade: Optional[float] = None
    regimes: Dict[str, RegimeConfig] = Field(default_factory=dict)
    parameters: Dict[str, float] = Field(default_factory=dict)


class RiskConfig(BaseModel):
    max_daily_loss: float = Field(default=1000.0)
    max_risk_per_trade: float = Field(default=50.0)
    max_open_risk: float = Field(default=0.0)
    max_trades_per_day: int = Field(default=0)
    max_trades_per_strategy: int = Field(default=0)
    max_open_positions: int = Field(default=5)
    max_positions_per_strategy: int = Field(default=3)
    max_notional_per_order: float = Field(default=5000.0)
    max_notional_per_symbol: float = Field(default=0.0)
    time_in_force: str = Field(default="day")
    risk_mode: str = Field(default="normal")

    # Map strategies by name
    strategies: Dict[str, StrategyConfig] = Field(default_factory=dict)
