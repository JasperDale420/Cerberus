from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ConvexityClass(str, Enum):
    """Strategy convexity classification for antifragile regime overrides."""

    CONVEX = "convex"
    LINEAR = "linear"
    CONCAVE = "concave"


class TrailingStopConfig(BaseModel):
    """Configuration for trailing stop behavior."""

    enabled: bool = False
    trail_pct: float = Field(default=0.02, description="Trail by X% from high water mark")
    min_profit_to_activate: float = Field(default=0.005, description="Only start trailing after X% profit")


class PartialExitConfig(BaseModel):
    """Configuration for partial profit taking."""

    enabled: bool = False
    first_exit_at_r: float = Field(default=1.0, description="Exit first tranche at X*R profit")
    first_exit_pct: float = Field(default=0.5, description="Exit X% of position at first target")


class AdvancedExitsConfig(BaseModel):
    """
    Advanced exit configuration for trailing stops, regime-aware stops, and partial exits.

    When enabled, broker-managed exits are disabled and PositionManager handles all exits.
    """

    enabled: bool = Field(default=False, description="Enable advanced exit management")
    trailing_stop: TrailingStopConfig = Field(default_factory=TrailingStopConfig)
    partial_exits: PartialExitConfig = Field(default_factory=PartialExitConfig)
    regime_aware_stops: bool = Field(default=True, description="Adjust stop width based on volatility regime")
    regime_stop_multipliers: Dict[str, float] = Field(
        default_factory=lambda: {
            "low": 0.75,
            "normal": 1.0,
            "high": 1.5,
            "shock": 2.0,
        },
        description="Stop width multipliers by volatility regime",
    )


class RegimeConfig(BaseModel):
    enabled: bool = True
    max_risk_per_trade: Optional[float] = None
    stop_multiplier: Optional[float] = None
    take_profit_multiplier: Optional[float] = None


class StrategyConfig(BaseModel):
    enabled: bool = True
    max_risk_per_trade: Optional[float] = None
    convexity_class: str = Field(default="linear", description="Convexity class: convex, linear, concave")
    regimes: Dict[str, RegimeConfig] = Field(default_factory=dict)
    parameters: Dict[str, float] = Field(default_factory=dict)


class PairTradingConfig(BaseModel):
    """Configuration for Pair Trading (StatArb)."""

    enabled: bool = False
    min_correlation: float = Field(default=0.8, description="Minimum Pearson correlation")
    max_eg_pvalue: float = Field(default=0.05, description="Maximum Engle-Granger p-value")
    lookback_days: int = Field(default=120, description="Lookback bars for cointegration test")
    entry_zscore: float = Field(default=2.0, description="Spread Z-Score to trigger entry")
    exit_zscore: float = Field(default=0.0, description="Spread Z-Score to trigger exit")
    max_active_pairs: int = Field(default=10, description="Maximum simultaneous active pairs")
    min_half_life: float = Field(default=2.0, description="Minimum mean-reversion half-life")
    max_half_life: float = Field(default=40.0, description="Maximum mean-reversion half-life")


class KellyConfig(BaseModel):
    """Kelly Criterion position sizing configuration."""

    enabled: bool = Field(default=False, description="Enable Kelly-based position sizing")
    fraction: float = Field(default=0.5, description="Kelly scalar: 1.0 = full Kelly, 0.5 = half-Kelly")
    lookback_trades: int = Field(default=50, description="Rolling window of trades for Kelly calculation")
    min_trades: int = Field(
        default=20, description="Minimum trades before Kelly activates (falls back to max_equity_pct)"
    )
    max_equity_pct: float = Field(default=0.30, description="Upper bound on Kelly-computed equity fraction")
    min_equity_pct: float = Field(default=0.05, description="Lower bound on Kelly-computed equity fraction")


class CPPIConfig(BaseModel):
    """CPPI drawdown-controlled position sizing configuration."""

    enabled: bool = Field(default=False, description="Enable CPPI drawdown control")
    multiplier: float = Field(default=3.0, description="CPPI multiplier (3-5 typical)")
    max_drawdown: float = Field(default=0.05, description="Maximum drawdown before floor locks (5%)")
    min_exposure: float = Field(default=0.1, description="Minimum exposure fraction (never go below 10%)")
    max_exposure: float = Field(default=1.0, description="Maximum exposure fraction")
    floor_decay_rate: float = Field(default=0.0, description="Daily floor decay rate (0 = no decay, ratchet only up)")


class HRPConfig(BaseModel):
    """Hierarchical Risk Parity cross-strategy allocation configuration."""

    enabled: bool = Field(default=False, description="Enable HRP cross-strategy allocation")
    lookback_days: int = Field(default=60, description="Rolling window for strategy returns")
    min_strategies: int = Field(default=3, description="Minimum active strategies to run HRP")
    rebalance_interval: int = Field(default=5, description="Recompute weights every N days")
    min_weight: float = Field(default=0.05, description="Minimum per-strategy weight")
    max_weight: float = Field(default=0.40, description="Maximum per-strategy weight")


class MomentumCrashConfig(BaseModel):
    """Momentum crash hedging configuration (Daniel & Moskowitz 2016)."""

    enabled: bool = Field(default=False, description="Enable momentum crash exposure reduction")
    min_exposure: float = Field(default=0.2, description="Minimum exposure floor during crash")
    bear_threshold: float = Field(default=-0.10, description="Cumulative return threshold for bear flag")
    spread_lookback: int = Field(default=60, description="Lookback bars for credit spread z-score")


class IVSurfaceConfigModel(BaseModel):
    """IV Surface Dynamics configuration (Breeden-Litzenberger 1978)."""

    enabled: bool = Field(default=False, description="Enable IV surface analysis on regime snapshot")
    skew_lookback: int = Field(default=20, description="Rolling window for skew z-score")
    skew_sigma_threshold: float = Field(default=2.0, description="Sigma threshold for skew steepening")
    min_strikes_for_rnd: int = Field(default=10, description="Minimum strikes for risk-neutral density")


class ETFFlowConfigModel(BaseModel):
    """Cross-Asset ETF Flow Propagation configuration."""

    enabled: bool = Field(default=False, description="Enable ETF flow divergence on regime snapshot")
    lookback_bars: int = Field(default=60, description="Rolling window for z-score computation")
    min_bars: int = Field(default=20, description="Minimum observations before producing signals")
    divergence_sigma: float = Field(default=2.0, description="Divergence significance threshold")
    etf_symbols: List[str] = Field(
        default_factory=lambda: ["SPY", "QQQ", "IWM"],
        description="ETF symbols to track for flow analysis",
    )


class CVaRConfig(BaseModel):
    """CVaR tail-risk position sizing configuration."""

    enabled: bool = Field(default=False, description="Enable CVaR-based position sizing")
    confidence_level: float = Field(default=0.95, description="CVaR confidence level (e.g., 0.95 = 95%)")
    lookback_bars: int = Field(default=500, description="Rolling window of bars for return distribution")
    min_bars: int = Field(default=100, description="Minimum bars before CVaR activates")
    max_acceptable_cvar: float = Field(default=0.03, description="Maximum acceptable CVaR per position (3%)")
    use_evt: bool = Field(default=True, description="Use Extreme Value Theory (GPD) for tail estimation")
    tail_threshold_pct: float = Field(default=10.0, description="Top X% of losses for EVT fitting")


class RiskConfig(BaseModel):
    max_daily_loss: float = Field(default=1000.0)
    max_risk_per_trade: float = Field(default=50.0)
    max_open_risk: float = Field(default=0.0)
    max_trades_per_day: int = Field(default=0)
    max_trades_per_strategy: int = Field(default=0)
    # P2.1 fix: Add max_orders_per_day to Pydantic model
    max_orders_per_day: int = Field(default=100)
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
            "complexity": {
                "structured": 1.20,
                "normal": 1.00,
                "random": 0.60,
            },
        },
        description="Multipliers applied to position sizing based on regime axes",
    )

    # Advanced exits configuration (trailing stops, partial exits, regime-aware stops)
    advanced_exits: AdvancedExitsConfig = Field(default_factory=AdvancedExitsConfig)

    # Tournament-style rank-based risk scaling
    alpha_rank_multipliers: Dict[int, float] = Field(
        default_factory=lambda: {
            1: 1.5,
            2: 1.25,
            3: 1.1,
            5: 1.0,
            10: 0.75,
        },
        description="Multipliers applied to max_risk_per_trade based on alpha rank",
    )

    pair_trading: PairTradingConfig = Field(default_factory=PairTradingConfig)

    # Kelly Criterion position sizing
    kelly: KellyConfig = Field(default_factory=KellyConfig)

    # CPPI drawdown-controlled sizing
    cppi: CPPIConfig = Field(default_factory=CPPIConfig)

    # HRP cross-strategy allocation
    hrp: HRPConfig = Field(default_factory=HRPConfig)

    # CVaR tail risk sizing
    cvar: CVaRConfig = Field(default_factory=CVaRConfig)

    # Momentum crash hedging (Daniel & Moskowitz 2016)
    momentum_crash: MomentumCrashConfig = Field(default_factory=MomentumCrashConfig)

    # IV Surface Dynamics (regime enrichment)
    iv_surface: IVSurfaceConfigModel = Field(default_factory=IVSurfaceConfigModel)

    # ETF Flow Propagation (regime enrichment)
    etf_flow: ETFFlowConfigModel = Field(default_factory=ETFFlowConfigModel)

    # Antifragile per-class regime overrides
    antifragile_overrides: Dict[str, Dict[str, Dict[str, float]]] = Field(
        default_factory=lambda: {
            "convex": {
                "vol": {"low": 0.80, "normal": 1.00, "high": 1.30, "shock": 1.50},
                "risk": {"risk_on": 1.00, "neutral": 1.00, "risk_off": 1.20},
            },
            "concave": {
                "vol": {"low": 1.10, "normal": 1.00, "high": 0.40, "shock": 0.00},
                "risk": {"risk_on": 1.00, "neutral": 0.80, "risk_off": 0.30},
            },
        },
        description="Per-convexity-class regime multiplier overrides (linear uses global defaults)",
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
