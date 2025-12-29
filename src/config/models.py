from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


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
    max_notional_per_order: float = Field(default=0.0)  # Fixed $ limit (0 = disabled)
    max_notional_pct: float = Field(default=0.05)  # % of equity (5% default)
    max_notional_per_symbol: float = Field(default=0.0)
    time_in_force: str = Field(default="day")
    risk_mode: str = Field(default="normal")

    # Map strategies by name
    strategies: Dict[str, StrategyConfig] = Field(default_factory=dict)

    # PRD Addendum: Regime-based risk multipliers
    regime_risk_multipliers: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "vol": {
                "low": 1.10,
                "normal": 1.00,
                "high": 0.60,
                "shock": 0.00,
            },
            "liquidity": {
                "good": 1.00,
                "thin": 0.75,
                "stressed": 0.00,
            },
            "risk": {
                "risk_on": 1.00,
                "neutral": 0.85,
                "risk_off": 0.50,
            },
        },
        description="Multipliers applied to position sizing based on regime axes",
    )

    # L5 fix: Bounds validation for critical risk parameters
    @field_validator("max_daily_loss")
    @classmethod
    def validate_max_daily_loss(cls, v: float) -> float:
        """Ensure max_daily_loss is positive and reasonable (max $100k)."""
        if v < 0:
            raise ValueError("max_daily_loss must be non-negative")
        if v > 100000:
            raise ValueError("max_daily_loss exceeds maximum allowed ($100,000)")
        return v

    @field_validator("max_risk_per_trade")
    @classmethod
    def validate_max_risk_per_trade(cls, v: float) -> float:
        """Ensure max_risk_per_trade is positive and reasonable (max $10k)."""
        if v < 0:
            raise ValueError("max_risk_per_trade must be non-negative")
        if v > 10000:
            raise ValueError("max_risk_per_trade exceeds maximum allowed ($10,000)")
        return v

    @field_validator("max_open_positions")
    @classmethod
    def validate_max_open_positions(cls, v: int) -> int:
        """Ensure max_open_positions is reasonable (1-100)."""
        if v < 0:
            raise ValueError("max_open_positions must be non-negative")
        if v > 100:
            raise ValueError("max_open_positions exceeds maximum allowed (100)")
        return v

    @field_validator("risk_mode")
    @classmethod
    def validate_risk_mode(cls, v: str) -> str:
        """Ensure risk_mode is one of the valid options."""
        valid_modes = ("normal", "reduced", "off")
        if v.lower() not in valid_modes:
            raise ValueError(f"risk_mode must be one of: {valid_modes}")
        return v.lower()
