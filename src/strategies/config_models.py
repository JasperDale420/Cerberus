"""
Pydantic configuration models for all trading strategies.

This module provides type-safe configuration classes for each strategy,
eliminating duplicate config.get() patterns and providing compile-time validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from src.engine.strategy_engine import StrategyActivationPolicy

# P3 fix: Constants for repeated field descriptions
_DESC_RISK_REWARD = "Risk:reward ratio"
_DESC_VOL_MULT = "Volume multiplier for confirmation"


class ActivationConfig(BaseModel):
    """
    Configuration for strategy activation based on multi-axis regimes.

    Parsed from strategies.yaml activation block:
    ```yaml
    activation:
      session: [opening, midday]
      trend: [flat]
      vol: [low, normal]
      min_confidence: 0.6
    ```
    """

    session: List[str] = Field(
        default_factory=list, description="Allowed session regimes"
    )
    trend: List[str] = Field(default_factory=list, description="Allowed trend regimes")
    vol: List[str] = Field(default_factory=list, description="Allowed vol regimes")
    liquidity: List[str] = Field(
        default_factory=list, description="Allowed liquidity regimes"
    )
    risk: List[str] = Field(default_factory=list, description="Allowed risk regimes")
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )

    def to_activation_policy(self) -> "StrategyActivationPolicy":
        """Convert to StrategyActivationPolicy with resolved enums."""
        from src.core.domain import (
            LiquidityRegime,
            RiskRegime,
            SessionRegime,
            TrendRegime,
            VolRegime,
        )
        from src.engine.strategy_engine import StrategyActivationPolicy

        def _resolve(values: List[str], enum_cls: type) -> Tuple[Any, ...]:
            """Resolve string values to enum members."""
            resolved = []
            for v in values:
                try:
                    resolved.append(enum_cls(v.lower()))
                except ValueError:
                    pass  # Skip invalid values
            return tuple(resolved)

        return StrategyActivationPolicy(
            session=_resolve(self.session, SessionRegime),
            trend=_resolve(self.trend, TrendRegime),
            vol=_resolve(self.vol, VolRegime),
            liquidity=_resolve(self.liquidity, LiquidityRegime),
            risk=_resolve(self.risk, RiskRegime),
            min_confidence=self.min_confidence,
        )


def build_activation_policies_from_config(
    strategies_config: Dict[str, Any],
) -> Dict[str, "StrategyActivationPolicy"]:
    """
    Build activation policies for all strategies from config dict.

    Args:
        strategies_config: The 'strategies' section from strategies.yaml

    Returns:
        Mapping of strategy name to StrategyActivationPolicy
    """
    policies: Dict[str, Any] = {}

    for name, params in strategies_config.items():
        if not isinstance(params, dict):
            continue

        activation_data = params.get("activation")
        if activation_data and isinstance(activation_data, dict):
            try:
                config = ActivationConfig(**activation_data)
                policies[name] = config.to_activation_policy()
            except Exception:
                pass  # Skip invalid configs

    return policies


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
    risk_reward: float = Field(default=2.0, gt=0, description=_DESC_RISK_REWARD)


class FlowMomentumConfig(BaseStrategyConfig):
    """Configuration for Flow-Confirmed Momentum strategy."""

    min_flow_zscore: float = Field(
        default=3.0, description="Minimum flow z-score threshold"
    )
    vol_mult: float = Field(default=1.5, gt=0, description=_DESC_VOL_MULT)
    risk_reward: float = Field(default=2.0, gt=0, description=_DESC_RISK_REWARD)


class GapFillConfig(BaseStrategyConfig):
    """Configuration for Gap Fill strategy."""

    min_gap: float = Field(
        default=0.015, ge=0, description="Minimum gap percentage (1.5%)"
    )
    max_gap: float = Field(
        default=0.10, ge=0, description="Maximum gap percentage (10%)"
    )
    risk_reward: float = Field(default=2.0, gt=0, description=_DESC_RISK_REWARD)
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
    risk_reward: float = Field(default=2.0, gt=0, description=_DESC_RISK_REWARD)
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
    risk_reward: float = Field(default=2.0, gt=0, description=_DESC_RISK_REWARD)
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
    risk_reward: float = Field(default=2.0, gt=0, description=_DESC_RISK_REWARD)
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
    risk_reward: float = Field(default=2.0, gt=0, description=_DESC_RISK_REWARD)
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
    vol_mult: float = Field(default=1.2, gt=0, description=_DESC_VOL_MULT)
    risk_reward: float = Field(default=2.0, gt=0, description=_DESC_RISK_REWARD)
    min_trend_score: float = Field(
        default=1.5, description="Minimum trend score threshold"
    )


class VixSpikeFadeConfig(BaseStrategyConfig):
    """Configuration for VIX Spike Fade strategy."""

    symbols: List[str] = Field(
        default=["SPY", "QQQ"], description="Allowed index symbols"
    )
    vix_spike_pct: float = Field(
        default=0.20, gt=0, description="VIX intraday spike threshold (20%)"
    )
    vix_absolute: float = Field(
        default=30.0, gt=0, description="Absolute VIX level trigger"
    )
    index_drop_pct: float = Field(
        default=0.015, gt=0, description="Minimum index decline from open (1.5%)"
    )
    reversion_target: float = Field(
        default=0.50, gt=0, le=1.0, description="Target % of decline to recover"
    )
    stop_buffer: float = Field(
        default=0.005, gt=0, description="Stop buffer below day low"
    )


class MomentumContinuationConfig(BaseStrategyConfig):
    """Configuration for Momentum Continuation strategy."""

    breakout_lookback: int = Field(
        default=5, ge=1, description="Days for high/low breakout levels"
    )
    vol_mult: float = Field(default=2.0, gt=0, description=_DESC_VOL_MULT)
    close_position: float = Field(
        default=0.75, gt=0, le=1.0, description="Required close position in bar range"
    )
    risk_reward: float = Field(default=1.5, gt=0, description=_DESC_RISK_REWARD)
    ema_fast: int = Field(default=20, ge=2, description="Fast EMA for trend")
    ema_slow: int = Field(default=50, ge=2, description="Slow EMA for trend")
    max_trades_per_session: int = Field(
        default=2, ge=1, description="Max trades per session"
    )
