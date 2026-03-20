import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

# --- Enums (PRD 3.1) ---


class Regime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    CHOP = "chop"


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class RiskMode(str, Enum):
    NORMAL = "normal"
    REDUCED = "reduced"
    OFF = "off"


# --- Multi-Axis Regime Enums (PRD Addendum) ---


class TrendRegime(str, Enum):
    """Trend direction axis."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class VolRegime(str, Enum):
    """Volatility state axis."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    SHOCK = "shock"


class LiquidityRegime(str, Enum):
    """Market liquidity axis."""

    GOOD = "good"
    THIN = "thin"
    STRESSED = "stressed"


class RiskRegime(str, Enum):
    """Risk appetite axis (derived from SPY/VXX dynamics)."""

    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


class SessionRegime(str, Enum):
    """Time-of-day session axis."""

    PREMARKET = "premarket"
    OPENING = "opening"
    MIDDAY = "midday"
    POWER_HOUR = "power_hour"
    CLOSE = "close"


class ComplexityRegime(str, Enum):
    """Market complexity regime derived from entropy metrics."""

    STRUCTURED = "structured"  # low entropy, predictable patterns forming
    NORMAL = "normal"  # typical market noise
    RANDOM = "random"  # high entropy, pure noise


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    """
    Multi-axis regime classification per PRD addendum.

    Replaces simple BULL/BEAR/CHOP with orthogonal axes:
    - trend: direction of market movement
    - vol: volatility state
    - liquidity: market depth quality
    - risk: risk-on/off sentiment
    - session: time-of-day

    Plus continuous features and per-axis confidence scores.
    """

    time: datetime
    index_symbol: str
    vol_symbol: Optional[str]

    # Discrete regime axes
    trend: TrendRegime
    vol: VolRegime
    liquidity: LiquidityRegime
    risk: RiskRegime
    session: SessionRegime

    # Continuous features (for logging/analytics)
    cum_ret: float = 0.0
    trend_strength: float = 0.0
    realized_vol: float = 0.0
    vol_of_vol: float = 0.0
    liquidity_score: float = 0.0
    risk_score: float = 0.0

    # BOCPD changepoint detection
    changepoint_probability: float = 0.0
    bars_since_changepoint: int = 0

    # Permutation entropy / complexity
    complexity: Optional["ComplexityRegime"] = None
    entropy_score: float = 0.0

    # Variance Risk Premium
    vrp_score: float = 0.0

    # GEX (Gamma Exposure) regime
    gex_regime: Optional[str] = None  # "positive", "negative", "neutral"
    net_gex: float = 0.0
    gamma_flip_distance: float = 0.0  # % distance to gamma flip level

    # IV Surface Dynamics (Breeden-Litzenberger 1978)
    iv_skew_zscore: float = 0.0
    iv_term_inverted: bool = False
    iv_signal_strength: float = 0.0

    # ETF Flow Propagation
    etf_flow_divergence: float = 0.0
    etf_rotation_detected: bool = False
    etf_lead_signal: float = 0.0

    # Per-axis confidence (0.0 to 1.0)
    confidence: Dict[str, float] = field(default_factory=dict)

    # Schema versioning for analytics
    model_version: str = "2.3"

    @property
    def regime_tags(self) -> Dict[str, str]:
        """Returns regime axes as string dict for logging/serialization."""
        return {
            "trend": self.trend.value,
            "vol": self.vol.value,
            "liquidity": self.liquidity.value,
            "risk": self.risk.value,
            "session": self.session.value,
        }

    @property
    def legacy_regime(self) -> "Regime":
        """Backwards-compatible mapping to legacy BULL/BEAR/CHOP."""
        if self.trend == TrendRegime.UP and self.trend_strength >= 0.65:
            return Regime.BULL
        if self.trend == TrendRegime.DOWN and self.trend_strength >= 0.65:
            return Regime.BEAR
        return Regime.CHOP


# --- Shared Core Types (PRD 3.2) ---


@dataclass
class Bar:
    symbol: str
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Optional fields for non-PRD extensions or computed values
    vwap: Optional[float] = None
    trade_count: Optional[int] = None


@dataclass
class Signal:
    symbol: str
    side: OrderSide  # "buy" or "sell" (for entry)
    size_hint: float  # suggested quantity (can be adjusted)
    entry_price: float  # reference, not necessarily order price
    stop_price: float
    target_price: float
    strategy: str
    generated_at: datetime
    meta: Dict[str, Any] = field(default_factory=dict)  # indicators, features, etc.
    correlation_id: str = ""  # for cross-module tracing
    # Legacy regime field kept for backward compatibility in older tests/callers.
    regime: Optional[Regime] = None
    # Multi-axis regime (replaces legacy BULL/BEAR/CHOP)
    regime_tags: Dict[str, str] = field(default_factory=dict)
    regime_confidence: Dict[str, float] = field(default_factory=dict)
    feature_snapshot: Optional["TechnicalFeatures"] = None  # Full alpha context

    def __post_init__(self) -> None:
        # PRD 3.2 / 2.1: correlation_id is required for cross-module tracing.
        # If a strategy does not supply one, derive a deterministic ID from the inputs.
        if not self.correlation_id:
            generated_at = self.generated_at
            if isinstance(generated_at, datetime):
                if generated_at.tzinfo is None:
                    generated_at = generated_at.replace(tzinfo=timezone.utc)
                epoch_ms = int(generated_at.timestamp() * 1000)
                iso_ts = generated_at.isoformat(timespec="microseconds")
            else:
                epoch_ms = 0
                iso_ts = "0"

            side_value = getattr(self.side, "value", str(self.side))
            canonical = (
                f"{self.strategy}|{self.symbol}|{side_value}|{iso_ts}|"
                f"{repr(self.entry_price)}|{repr(self.stop_price)}|{repr(self.target_price)}"
            )
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
            self.correlation_id = f"{self.strategy}-{self.symbol}-{epoch_ms}-{digest}"


@dataclass
class OrderIntent:
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    limit_price: Optional[float]
    time_in_force: str
    correlation_id: str
    strategy: str
    stop_loss: Optional[float]
    take_profit: Optional[float]
    meta: Dict[str, Any] = field(default_factory=dict)


# --- Symbol & Market State (PRD 3.3) ---


@dataclass
class Position:
    symbol: str
    side: Side
    qty: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    strategy: str
    entry_time: Optional[datetime] = None
    correlation_id: str = ""
    # Legacy regime field kept for backward compatibility in older tests/callers.
    regime_at_entry: Optional[Regime] = None
    # Multi-axis regime at entry (replaces legacy Regime enum)
    regime_tags_at_entry: Dict[str, str] = field(default_factory=dict)
    open_risk: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    entry_features: Optional[dict] = None
    mae_r: float = 0.0
    mfe_r: float = 0.0
    commission: float = 0.0
    slippage_estimate: float = 0.0
    max_hold_seconds: Optional[int] = None
    # H1 fix: Track when position last modified to prevent reconciliation race conditions
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Advanced Exit Fields ---
    # Trailing stop configuration
    trailing_stop_enabled: bool = False
    trailing_stop_pct: Optional[float] = None  # Trail by X% from high water mark
    trailing_high_water: Optional[float] = None  # Best favorable price seen
    initial_stop_price: Optional[float] = None  # Original stop before trailing

    # Partial exit tracking
    initial_qty: float = 0.0  # Original position size at entry
    partial_exits_taken: int = 0  # Number of partial exits completed
    partial_exit_levels: List[tuple] = field(default_factory=list)  # [(r_mult, fraction), ...]

    # Trail activation gate: min R-profit before trailing starts
    trail_min_profit_r: Optional[float] = None

    # Regime-aware stop multiplier applied at entry
    regime_stop_multiplier: float = 1.0

    # Bar timestamp at entry — used to skip exit checks on the entry bar
    entry_bar_time: Optional[datetime] = None

    # Explicit strategy name for per-strategy EOD logic (e.g. overnight hold decisions)
    strategy_name: str = ""


@dataclass
class SymbolState:
    symbol: str
    bars_1m: Deque[Bar]
    bars_5m: Deque[Bar]
    bars_15m: Deque[Bar]
    bars_1d: Deque[Bar]
    indicators: Dict[str, Any]
    position: Optional[Position]
    open_orders: Dict[str, Any]  # keyed by broker order ID
    allowed_strategies: List[str]  # set from scanner
    meta: Dict[str, Any]  # e.g. scanner_score, ATR

    def __init__(
        self,
        symbol: str,
        bars: Deque[Bar] = None,
        indicators: Dict[str, Any] = None,
        position: Optional[Position] = None,
        open_orders: Dict[str, Any] = None,
        allowed_strategies: List[str] = None,
        meta: Dict[str, Any] = None,
        bars_1m: Deque[Bar] = None,
        bars_5m: Deque[Bar] = None,
        bars_15m: Deque[Bar] = None,
        bars_1d: Deque[Bar] = None,
    ):
        from collections import deque

        self.symbol = symbol
        self.bars_1m = bars_1m if bars_1m is not None else (bars if bars is not None else deque(maxlen=24 * 60))
        self.bars_5m = bars_5m if bars_5m is not None else deque(maxlen=24 * 12)
        self.bars_15m = bars_15m if bars_15m is not None else deque(maxlen=8 * 12)
        self.bars_1d = bars_1d if bars_1d is not None else deque(maxlen=252)
        self.indicators = indicators if indicators is not None else {}
        self.position = position
        self.open_orders = open_orders if open_orders is not None else {}
        self.allowed_strategies = allowed_strategies if allowed_strategies is not None else []
        self.meta = meta if meta is not None else {}

    @property
    def bars(self) -> Deque[Bar]:
        return self.bars_1m

    @bars.setter
    def bars(self, value: Deque[Bar]) -> None:
        self.bars_1m = value


@dataclass
class MarketState:
    time: datetime
    regime: Regime  # Legacy: kept for backwards compatibility
    index_symbol: str = "SPY"
    index_price: float = 0.0
    index_return: float = 0.0
    realized_vol: float = 0.0
    daily_pnl: float = 0.0
    risk_mode: RiskMode = RiskMode.NORMAL
    meta: Dict[str, Any] = field(default_factory=dict)
    # PRD Addendum: Full multi-axis regime snapshot
    regime_snapshot: Optional[MarketRegimeSnapshot] = None


@dataclass
class TechnicalFeatures:
    """
    Standard container for calculated features (replacing raw tuples).
    See: src/data/calculator.py
    """

    price: float
    volume: float
    timestamp: datetime
    atr: float
    atr_pct: float
    intraday_range_pct: float
    gap_pct: float
    ema20_slope: float
    distance_from_vwap: float
    adx: float
    distance_from_ema20: float
    prior_day_high: float
    prior_day_low: float
    bb_upper: float
    bb_lower: float
    price_zscore: float
    premarket_volume: float
    last_updated: datetime

    # New Mean Reversion factors for ranking
    ma_dist_50: float = 0.0
    ma_dist_200: float = 0.0
    tfi: float = 0.0  # Trade Flow Imbalance (-1.0 to 1.0)
    net_gex: float = 0.0  # Total Net Gamma Exposure (MM hedging pressure)
    gex_flip_dist: float = 0.0  # Distance to GEX Flip/Pin level (%)
    frac_diff_close: float = 0.0  # Fractionally differentiated close price
    hurst_exponent: float = 0.0  # Hurst Exponent (0-1, <0.5 MR, >0.5 Trend)
    ofi: float = 0.0  # Order Flow Imbalance (proxy for quote imbalances)
    high_low_spread: float = 0.0  # Daily bar spread scaled by price

    # Fusion Alpha Fields (with defaults, MUST be at the end)
    relative_strength: float = 0.0  # Stock return vs Benchmark (e.g. SPY)
    flow_bias: float = 0.0  # Institutional net direction score (-1 to 1)
    dof_score: float = 0.0  # Directional Options Flow score (0 to 1)
    orb_high: float = 0.0  # Opening Range High (9:30-9:45 AM ET)
    orb_low: float = 0.0  # Opening Range Low (9:30-9:45 AM ET)


# --- Scanner / Features (PRD 4.3) ---


@dataclass
class SymbolFeatures:
    symbol: str
    price: float
    atr_pct: float
    avg_volume: float
    intraday_range_pct: float
    gap_pct: float
    ema20_slope: float
    ema_trend_strength: float
    distance_from_vwap: float
    premarket_volume: float
    adx: float  # Trend Strength
    distance_from_ema20: float  # (Price - EMA20) / EMA20

    # Key Levels
    prior_day_high: float
    prior_day_low: float

    # Mean Reversion
    bb_upper: float
    bb_lower: float
    price_zscore: float

    # options flow (Unusual Whales)
    flow_zscore: float
    call_put_ratio: float
    large_sweeps_count: int
    aggressive_flow_share: float
    last_updated: datetime

    # Fusion Signals
    relative_strength: float = 0.0
    flow_bias: float = 0.0
    dof_score: float = 0.0
    atr: float = 0.0
    orb_high: float = 0.0
    orb_low: float = 0.0
    tfi: float = 0.0
    net_gex: float = 0.0
    gex_flip_dist: float = 0.0
    frac_diff_close: float = 0.0
    hurst_exponent: float = 0.0

    # Ranking Engine Outputs
    ma_dist_50: float = 0.0
    ma_dist_200: float = 0.0
    alpha_score: float = 0.0
    alpha_rank: int = 0

    # Microstructure Features
    ofi: float = 0.0  # Order Flow Imbalance (proxy for quote imbalances)
    high_low_spread: float = 0.0  # Daily bar spread scaled by price

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineInfo:
    """
    PRD 4.3: baseline filter output structure.
    """

    symbol: str
    last_price: float
    avg_volume: float
    atr_pct: float


@dataclass
class StrategyCandidate:
    """
    PRD 4.5: per-strategy candidate before symbol grouping.
    """

    symbol: str
    strategy: str
    score: float
    features: SymbolFeatures


@dataclass
class WatchlistSymbol:
    symbol: str
    score: float
    strategies: List[str]
    features: SymbolFeatures


@dataclass
class ScanResult:
    generated_at: datetime
    regime: Regime
    watchlist: List[WatchlistSymbol]

    @property
    def symbol_map(self) -> Dict[str, WatchlistSymbol]:
        return {w.symbol: w for w in self.watchlist}
