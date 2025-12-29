"""
Pydantic configuration models for all trading strategies.

This module provides type-safe configuration classes for each strategy,
eliminating duplicate config.get() patterns and providing compile-time validation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BaseStrategyConfig(BaseModel):
    """Base configuration for all strategies."""

    cooldown_bars: int = Field(
        default=5, ge=0, description="Minimum bars between signals"
    )

    class Config:
        """Pydantic config."""

        extra = "allow"  # Allow extra fields for backward compatibility


class FailedBreakoutConfig(BaseStrategyConfig):
    """Configuration for Failed Breakout (Fade) strategy."""

    lookback_days: int = Field(
        default=1, ge=1, description="Days to look back for high/low"
    )
    risk_reward: float = Field(default=2.0, gt=0, description="Risk:reward ratio")


class FlowMomentumConfig(BaseStrategyConfig):
    """Configuration for Flow-Confirmed Momentum strategy."""

    min_flow_zscore: float = Field(
        default=3.0, description="Minimum flow z-score threshold"
    )
    vol_mult: float = Field(
        default=1.5, gt=0, description="Volume multiplier for confirmation"
    )
    risk_reward: float = Field(default=2.0, gt=0, description="Risk:reward ratio")


class GapFillConfig(BaseStrategyConfig):
    """Configuration for Gap Fill strategy."""

    min_gap: float = Field(
        default=0.015, ge=0, description="Minimum gap percentage (1.5%)"
    )
    max_gap: float = Field(
        default=0.10, ge=0, description="Maximum gap percentage (10%)"
    )
    risk_reward: float = Field(default=2.0, gt=0, description="Risk:reward ratio")
    or_time_minutes: int = Field(
        default=15, ge=1, description="Opening range time window in minutes"
    )
    weak_trend_max_score: float = Field(
        default=1.0, ge=0, description="Max trend score for entry (weak trend filter)"
    )


class IndexMeanReversionConfig(BaseStrategyConfig):
    """Configuration for Index Mean Reversion strategy."""

    bb_len: int = Field(default=20, ge=2, description="Bollinger Bands length")
    bb_std: float = Field(
        default=2.0, gt=0, description="Bollinger Bands standard deviations"
    )
    risk_reward: float = Field(default=2.0, gt=0, description="Risk:reward ratio")
    stop_std: float = Field(
        default=3.0, gt=0, description="Stop loss in standard deviations"
    )
    stop_pct: float = Field(
        default=0.005, gt=0, description="Stop loss as percentage (0.5%)"
    )
    symbols: List[str] = Field(
        default=["SPY", "QQQ"], description="Allowed index ETF symbols"
    )

    @field_validator("symbols", mode="before")
    @classmethod
    def normalize_symbols(cls, v):
        """Normalize symbols to uppercase."""
        if isinstance(v, list):
            return [str(s).upper() for s in v]
        return v


class ORBConfig(BaseStrategyConfig):
    """Configuration for Opening Range Breakout strategy."""

    orb_minutes: int = Field(
        default=15, ge=1, description="Opening range duration in minutes"
    )
    risk_reward: float = Field(default=2.0, gt=0, description="Risk:reward ratio")
    stop_loss_pct: float = Field(
        default=0.005, gt=0, description="Stop loss percentage (0.5%)"
    )
    min_gap_pct: float = Field(
        default=0.0, ge=0, description="Minimum gap percentage filter"
    )
    min_flow_zscore: float = Field(
        default=0.0, description="Minimum flow z-score filter"
    )
    min_premarket_volume: float = Field(
        default=0.0, ge=0, description="Minimum premarket volume filter"
    )
    # P2 fix: ATR-based buffer for stop placement
    stop_buffer_atr_mult: float = Field(
        default=0.0, ge=0, description="ATR multiplier for stop buffer (0=disabled)"
    )


class TrendPullbackConfig(BaseStrategyConfig):
    """Configuration for Trend Pullback strategy."""

    ema_fast: int = Field(default=20, ge=2, description="Fast EMA period")
    ema_slow: int = Field(default=50, ge=2, description="Slow EMA period")
    rsi_len: int = Field(default=2, ge=1, description="RSI period")
    risk_reward: float = Field(default=2.0, gt=0, description="Risk:reward ratio")
    pullback_depth_pct: float = Field(
        default=0.0, ge=0, description="Maximum pullback depth percentage"
    )
    entry_confirmation: str = Field(
        default="rsi", description="Entry confirmation method (rsi, none)"
    )
    rsi_oversold: float = Field(
        default=10, ge=0, le=100, description="RSI oversold threshold"
    )
    rsi_overbought: float = Field(
        default=90, ge=0, le=100, description="RSI overbought threshold"
    )
    # P3 fix: Configurable stop lookback instead of hardcoded 3
    stop_lookback_bars: int = Field(
        default=5, ge=1, description="Number of bars for stop placement"
    )


class VWAPReversionConfig(BaseStrategyConfig):
    """Configuration for VWAP Reversion strategy."""

    sigma_band: Optional[float] = Field(
        default=None, description="Sigma band (PRD naming, overrides band_sigma)"
    )
    band_sigma: float = Field(default=2.0, gt=0, description="VWAP band sigma")
    risk_reward: float = Field(default=2.0, gt=0, description="Risk:reward ratio")
    time_window_start: str = Field(
        default="09:45", description="Trading window start time (HH:MM)"
    )
    time_window_end: str = Field(
        default="15:45", description="Trading window end time (HH:MM)"
    )
    max_hold_minutes: int = Field(
        default=60, ge=1, description="Maximum hold time in minutes"
    )
    confirmation: str = Field(
        default="rsi", description="Confirmation method (rsi, none)"
    )
    rsi_len: int = Field(default=2, ge=1, description="RSI period")
    rsi_oversold: float = Field(
        default=10, ge=0, le=100, description="RSI oversold threshold"
    )
    rsi_overbought: float = Field(
        default=90, ge=0, le=100, description="RSI overbought threshold"
    )

    @property
    def effective_band_sigma(self) -> float:
        """Get effective band sigma (sigma_band takes precedence)."""
        return self.sigma_band if self.sigma_band is not None else self.band_sigma


class VWAPTrendRiderConfig(BaseStrategyConfig):
    """Configuration for VWAP Trend Rider strategy."""

    ema_fast: int = Field(default=20, ge=2, description="Fast EMA period")
    ema_slow: int = Field(default=50, ge=2, description="Slow EMA period")
    vol_mult: float = Field(
        default=1.2, gt=0, description="Volume multiplier for confirmation"
    )
    risk_reward: float = Field(default=2.0, gt=0, description="Risk:reward ratio")
    min_trend_score: float = Field(
        default=1.5, description="Minimum trend score threshold"
    )
