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
    regime: Regime
    generated_at: datetime
    meta: Dict[str, Any] = field(default_factory=dict)  # indicators, features, etc.
    correlation_id: str = ""  # for cross‑module tracing

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
    regime_at_entry: Optional[Regime] = None
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


@dataclass
class SymbolState:
    symbol: str
    bars: Deque[Bar]
    indicators: Dict[str, Any]
    position: Optional[Position]
    open_orders: Dict[str, Any]  # keyed by broker order ID
    allowed_strategies: List[str]  # set from scanner
    meta: Dict[str, Any]  # e.g. scanner_score, ATR


@dataclass
class MarketState:
    time: datetime
    regime: Regime
    index_symbol: str = "SPY"
    index_price: float = 0.0
    index_return: float = 0.0
    realized_vol: float = 0.0
    daily_pnl: float = 0.0
    risk_mode: RiskMode = RiskMode.NORMAL
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TechnicalFeatures:
    """
    Standard container for calculated features (replacing raw tuples).
    See: src/data/calculator.py
    """

    price: float
    volume: float
    timestamp: datetime
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

    # misc
    last_updated: datetime
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
