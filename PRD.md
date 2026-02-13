PRD – Multi‑Strategy Intraday Scalping System (Equities, Alpaca + Unusual Whales)

0. Overview

Build a deterministic, intraday, multi‑strategy scalping system for US equities using:
	•	Alpaca (REST + WebSocket) for data and order routing
	•	Unusual Whales for options‑flow‑based signal features

Key properties:
	•	Vertical slice architecture: the system must be implementable and testable end‑to‑end in small, independently working slices (e.g., “one strategy, one symbol, full pipeline”).
	•	Deterministic behavior: given the same inputs and config, the system must make the same decisions.
	•	Robust error logging: consistent, structured logging and error handling across all modules.
	•	Agent‑driven continuous improvement: an offline “agent” that reviews trades, updates configuration, and (later) proposes code changes based on performance.

Initial scope: equities only, intraday scalping (no overnight holds). Options support is a future extension.

⸻

1. Goals & Success Criteria

1.1 Primary Goals
	1.	Trade multiple equities simultaneously (up to Alpaca’s ~30 ticker WebSocket limit).
	2.	Use a scanner (REST + Unusual Whales) to select and prioritize tradeable symbols.
	3.	Use a WebSocket execution engine to:
	•	Monitor selected symbols in real time
	•	Run multiple plug‑and‑play strategies per symbol
	•	Route orders via Alpaca with risk controls
	4.	Maintain a market regime classifier (bull / bear / chop) to select regime‑appropriate strategies.
	5.	Log all signals, orders, trades, and context for post‑trade analytics.
	6.	Run an end‑of‑day Agent that:
	•	Evaluates performance per strategy/regime
	•	Adjusts configuration automatically within safe bounds
	•	(Later) proposes code changes for human review

1.2 Success Metrics
	•	System can run a complete vertical slice:
	•	Single strategy (e.g., VWAP Reversion)
	•	Single symbol
	•	Full pipeline: scanner → execution engine → risk → orders → logs → analytics → agent config adjustment.
	•	Stable intraday operation (no crashes, graceful degradation on errors).
	•	Per‑strategy performance stats available for at least:
	•	PnL (gross/net, in R)
	•	Winrate
	•	Expectancy
	•	Drawdown
	•	Agent can disable underperforming strategy–regime pairs and tighten risk without human intervention.

⸻

2. High‑Level Architecture

2.1 Components
	1.	Config Layer
	•	YAML/JSON configs for:
	•	Strategies (strategies.yaml)
	•	Risk (risk.yaml)
	•	Scanner (scanner.yaml)
	•	Universe (universe.yaml)
	•	Logging (logging.yaml)
	•	Agent‑generated overrides (e.g., strategies.auto.yaml).
	2.	Scanner & Data Layer
	•	Universe builder (equity universe).
	•	Baseline filters for liquidity/price.
	•	Feature pipeline (price, volatility, technicals, options flow).
	•	Strategy‑aware scoring.
	•	Watchlist builder (≤ 30 symbols).
	3.	Market Regime Detector
	•	SPY (or another index)–based intraday classifier:
	•	BULL, BEAR, CHOP
	•	Uses rolling return vs volatility with smoothing.
	4.	Execution Engine
	•	Alpaca WebSocket listener for bars and trading events.
	•	Maintains:
	•	SymbolState for each symbol
	•	MarketState (including regime)
	•	Uses:
	•	StrategyEngine → signals
	•	RiskManager → OrderIntents
	•	OrderExecutor → Alpaca orders
	•	PositionManager → exits and PnL
	5.	Strategy Layer
	•	Base BaseStrategy interface.
	•	Multiple plug‑in strategies:
	•	VWAP Reversion
	•	Trend Pullback
	•	Opening Range Breakout/Breakdown (ORB)
	•	Failed Breakout Fade
	•	VWAP Trend Rider
	•	Index Mean Reversion
	•	Flow‑Confirmed Momentum
	•	Gap‑Fill
	•	Mapped to regimes and symbols via config and scanner output.
	6.	Analytics Layer
	•	DB schema with tables:
	•	trades
	•	signals
	•	orders
	•	fills
	•	regime_history
	•	scanner_snapshots
	•	strategy_stats_daily
	•	agent_actions
	•	Reporting and metrics computation.
	7.	Agent Layer
	•	Nightly process:
	•	Stage 1: deterministic strategy health + risk adjustments.
	•	Stage 2: parameter tuning (offline backtests or replay).
	•	Stage 3: code‑level proposals (with human review).
	•	Writes config overrides and produces human‑readable reports.
	8.	Logging & Error Handling
	•	Centralized structured logging and error handling across modules.
	•	Correlation IDs across signal → order → trade.
	•	Standard log levels and error taxonomy.

⸻

3. Domain Model & Core Types

Use Python dataclasses & typing.

3.1 Enums

from enum import Enum

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

3.2 Shared Core Types

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List

@dataclass
class Bar:
    symbol: str
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    symbol: str
    side: OrderSide            # "buy" or "sell" (for entry)
    size_hint: float           # suggested quantity (can be adjusted)
    entry_price: float         # reference, not necessarily order price
    stop_price: float
    target_price: float
    strategy: str
    regime: Regime
    generated_at: datetime
    meta: Dict[str, Any]       # indicators, features, etc.
    correlation_id: str        # for cross‑module tracing


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
    meta: Dict[str, Any]

3.3 Symbol & Market State

@dataclass
class Position:
    symbol: str
    side: Side
    qty: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    strategy: str


@dataclass
class SymbolState:
    symbol: str
    bars: Deque[Bar]
    indicators: Dict[str, Any]
    position: Optional[Position]
    open_orders: Dict[str, Any]         # keyed by broker order ID
    allowed_strategies: List[str]       # set from scanner
    meta: Dict[str, Any]                # e.g. scanner_score, ATR


@dataclass
class MarketState:
    time: datetime
    regime: Regime
    index_symbol: str
    index_price: float
    index_return: float
    realized_vol: float
    daily_pnl: float
    risk_mode: RiskMode
    meta: Dict[str, Any]


⸻

4. Scanner & Data Layer

4.1 Responsibilities
	•	Build a base universe of equities.
	•	Apply baseline filters for liquidity, price, volatility.
	•	Compute feature vectors for each symbol:
	•	Price/volatility
	•	Liquidity
	•	Technicals
	•	Options flow (Unusual Whales)
	•	Use strategy‑specific scoring to nominate symbols.
	•	Build a global watchlist (≤ 30 symbols) that includes:
	•	Symbol
	•	Score
	•	Allowed strategies
	•	Feature snapshot
	•	Periodically re‑run & update execution engine watchlist with throttled churn.

4.2 Universe Builder

Config: universe.yaml
	•	Sources:
	•	Static: S&P500, NASDAQ100 lists.
	•	Dynamic: top volume by previous day, etc.
	•	Explicit symbols.

Interface:

class UniverseBuilder:
    def __init__(self, config, alpaca_client):
        ...

    def build_universe(self) -> List[str]:
        """
        Combine static lists + dynamic filters.
        Returns a list of symbols.
        """

4.3 Baseline Filters & Features

Baseline filters (configurable):
	•	Price range: e.g., 3 ≤ price ≤ 300.
	•	Avg daily volume (last N days) ≥ threshold.
	•	Exclude extremely illiquid or penny stocks.

Output structure:

@dataclass
class BaselineInfo:
    symbol: str
    last_price: float
    avg_volume: float
    atr_pct: float

Features (per symbol):

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

    # Trend & Mean Reversion
    adx: float                 # Trend Strength
    distance_from_ema20: float # (Price - EMA20) / EMA20
    bb_upper: float            # Bollinger Band Upper (20, 2)
    bb_lower: float            # Bollinger Band Lower (20, 2)
    price_zscore: float        # Z-Score of price vs 20-period mean/std

    # Key Levels
    prior_day_high: float
    prior_day_low: float

    # Options flow (Unusual Whales)
    flow_zscore: float
    call_put_ratio: float
    large_sweeps_count: int
    aggressive_flow_share: float

    # misc
    last_updated: datetime
    extra: Dict[str, Any]

FeaturePipeline:

class FeaturePipeline:
    def __init__(self, alpaca_client, whales_client, config, logger):
        ...

    def compute_features(self, symbols: List[str]) -> Dict[str, SymbolFeatures]:
        """
        For each symbol:
          - fetch price/volume data from Alpaca
          - fetch flow data from Unusual Whales
          - compute features
        On partial failures:
          - log error
          - mark flow features as neutral (e.g. 0) if flow unavailable
        """

4.4 Strategy Scanner Profiles

Each strategy defines how much it “likes” a symbol’s current conditions.

from abc import ABC, abstractmethod

class StrategyScannerProfile(ABC):
    name: str

    @abstractmethod
    def min_requirements(self, f: SymbolFeatures) -> bool:
        """Hard prerequisites; if false, symbol not considered for this strategy."""
        ...

    @abstractmethod
    def score(self, f: SymbolFeatures, regime: Regime) -> float:
        """
        Compute a float score. Higher = more attractive.
        Must be deterministic.
        """

Examples (high‑level):
	•	VWAP Reversion:
	•	Likes: moderate ATR, decent volume, distance_from_vwap > threshold, regime = CHOP.
	•	ORB Breakout:
	•	Likes: large gap_pct, high premarket_volume, high flow_zscore.

4.5 Watchlist Construction

Intermediate:

@dataclass
class StrategyCandidate:
    symbol: str
    strategy: str
    score: float
    features: SymbolFeatures

Final structure:

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

Algorithm:
	1.	Universe → baseline filters → features.
	2.	For each strategy and symbol:
	•	If min_requirements true, compute score.
	•	Keep top K candidates per strategy.
	3.	Group by symbol:
	•	symbol_score = max(score over strategies)
	•	strategies = list of strategies that nominated it.
	4.	Sort symbols by symbol_score.
	5.	Take top MAX_WATCHLIST_SIZE (≤ 30).

4.6 Scanner Orchestration

class Scanner:
    def __init__(self, universe_builder, feature_pipeline,
                 strategy_profiles, config, logger):
        ...

    def run_scan(self, regime: Regime) -> ScanResult:
        """
        Calculate ScanResult.
        All errors logged; partial results allowed (with degraded features).
        """

Periodic execution:
	•	Run at open + every X minutes.
	•	Results logged to scanner_snapshots (see Analytics).

⸻

5. Market Regime Detector

5.1 Inputs
	•	Real‑time SPY (or chosen index) bars (1‑minute recommended).
	•	Rolling window of last N bars (e.g., N=60).
	•	Minimum number of bars before classifying (e.g., 20).

5.2 Features

For each update:
	•	Returns: r_t = log(close_t / close_{t-1}).
	•	cum_ret = sum(r_t) over window.
	•	vol = std(r_t) + epsilon.
	•	trend_score = abs(cum_ret) / vol.

Optional (for debugging, stored in regime_history):
	•	ema20_slope
	•	r2 of linear regression of price vs time.

5.3 Classification

Core classifier:

def classify_regime(cum_ret: float, trend_score: float,
                    up_thresh: float = 1.5,
                    down_thresh: float = 1.5) -> Regime:
    if trend_score < 1.0:
        return Regime.CHOP
    if cum_ret > 0 and trend_score >= up_thresh:
        return Regime.BULL
    if cum_ret < 0 and trend_score >= down_thresh:
        return Regime.BEAR
    return Regime.CHOP

5.4 Smoothing / Hysteresis

Use a “majority vote” over last K classifications:

from collections import deque, Counter

class RegimeDetector:
    def __init__(self, window: int = 60, min_bars: int = 20,
                 smooth_k: int = 10, logger=None):
        self.spy_bars = deque(maxlen=window)
        self.last_classifications = deque(maxlen=smooth_k)
        self.current_regime = Regime.CHOP
        self.min_bars = min_bars
        self.logger = logger

    def update(self, bar: Bar) -> Regime:
        self.spy_bars.append(bar)
        if len(self.spy_bars) < self.min_bars:
            return self.current_regime

        try:
            cum_ret, trend_score = compute_regime_features(self.spy_bars)
            new_regime = classify_regime(cum_ret, trend_score)
        except Exception as e:
            # Log & keep previous regime
            self.logger.error("RegimeDetector.update failed",
                              extra={"error": str(e)})
            return self.current_regime

        self.last_classifications.append(new_regime)
        self.current_regime = self._smooth_regime()
        return self.current_regime

    def _smooth_regime(self) -> Regime:
        counts = Counter(self.last_classifications)
        top_regime, _ = max(counts.items(), key=lambda kv: kv[1])
        return top_regime

    def get_regime(self) -> Regime:
        return self.current_regime

All regime updates logged to regime_history.

⸻

6. Execution Engine

6.1 Responsibilities
	•	Subscribe to Alpaca stock data (bars) for:
	•	Index (SPY).
	•	Watchlist symbols.
	•	Maintain SymbolState and MarketState.
	•	On each bar:
	•	Update state.
	•	Run StrategyEngine to generate signals.
	•	Pass signals to RiskManager → OrderIntents.
	•	Send orders via OrderExecutor.
	•	Update positions via PositionManager.
	•	Log all events for analytics.

6.2 WebSocket Integration
	•	Use Alpaca’s live data stream (e.g., StockDataStream).
	•	Data handler calls ExecutionEngine.on_bar(bar).

6.3 Watchlist Management

Execution engine exposes:

class ExecutionEngine:
    def apply_scan_result(self, scan_result: ScanResult):
        """
        - Compare current symbol set vs new watchlist.
        - Subscribe/unsubscribe symbols (with churn limits).
        - Create/remove SymbolState entries.
        - SPY/index symbol remains subscribed always.
        """

Churn throttling:
	•	At most N symbols added/removed per scan interval.
	•	If more changes are required, roll changes across multiple intervals.

6.4 Strategy Engine

class StrategyEngine:
    def __init__(self, strategies_by_name: Dict[str, "BaseStrategy"],
                 strategies_by_regime: Dict[Regime, List[str]],
                 logger):
        self.strategies_by_name = strategies_by_name
        self.strategies_by_regime = strategies_by_regime
        self.logger = logger

    def on_bar(self,
               symbol: str,
               bar: Bar,
               symbol_state: SymbolState,
               market_state: MarketState) -> List[Signal]:
        regime = market_state.regime
        allowed = set(symbol_state.allowed_strategies)
        regime_strats = set(self.strategies_by_regime.get(regime, []))
        active_strats = sorted(allowed.intersection(regime_strats))

        signals: List[Signal] = []
        for name in active_strats:
            strat = self.strategies_by_name[name]
            try:
                sig = strat.on_bar(symbol, bar, symbol_state, market_state)
            except Exception as e:
                self.logger.error("Strategy error",
                                  extra={"strategy": name,
                                         "symbol": symbol,
                                         "error": str(e)})
                continue
            if sig:
                signals.append(sig)
        return signals

6.5 Risk Manager

Responsibilities:
	•	Enforce account‑level risk:
	•	Max daily loss
	•	Max open risk
	•	Enforce strategy‑level risk:
	•	Max R per trade
	•	Max trades per day
	•	Max concurrent positions per strategy
	•	Enforce symbol‑level risk:
	•	Max exposure per symbol
	•	Handle risk mode (NORMAL / REDUCED / OFF).

class RiskManager:
    def __init__(self, config, account_state_provider, logger):
        ...

    def apply(self,
              signal: Signal,
              symbol_state: SymbolState,
              market_state: MarketState) -> List[OrderIntent]:
        """
        - Check daily PnL vs limits.
        - Check max open risk.
        - Determine position size based on initial risk (entry→stop).
        - If rejected, log with reason.
        - If accepted, create one or more OrderIntent (entry + OCO).
        """

Logging:
	•	All accepted/rejected signals recorded in signals table with:
	•	accepted
	•	rejection_reason

6.6 Order Executor

Responsible for:
	•	Converting OrderIntent into Alpaca orders.
	•	Handling:
	•	submission
	•	order status updates
	•	cancellation logic
	•	Mapping Alpaca order IDs back to correlation IDs and trades.

class OrderExecutor:
    def __init__(self, alpaca_trading_client, logger):
        ...

    def submit(self, intent: OrderIntent):
        try:
            # translate intent → Alpaca order
            ...
        except Exception as e:
            self.logger.error("Order submission failed",
                              extra={"symbol": intent.symbol,
                                     "correlation_id": intent.correlation_id,
                                     "error": str(e)})
            # Optionally mark in DB as "order_failed"

6.7 Position Manager
	•	Tracks open positions per symbol and strategy.
	•	On bar:
	•	Check if price hit or crossed stop/target when those are not fully delegated to the broker.
	•	Create exit OrderIntents as needed.
	•	On fills:
	•	Update Position.
	•	Update unrealized_pnl, realized_pnl.
	•	Detect trade completion and trigger DB log.

⸻

7. Strategy Layer

7.1 Base Interface

from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger

    @abstractmethod
    def on_bar(self,
               symbol: str,
               bar: Bar,
               symbol_state: SymbolState,
               market_state: MarketState) -> Optional[Signal]:
        """
        Strategies must be deterministic given:
          - config
          - symbol_state
          - market_state
          - bar

        They may:
          - open new positions
          - add to position (if allowed)
          - signal exits (via side and sizes)
        """

7.2 Initial Strategy Set (High‑Level Specs)
	1.	VWAP Reversion Scalper (CHOP)
	•	Use intraday VWAP.
	•	Enter when price deviates beyond Nσ from VWAP and shows reversal behavior (candlestick or short‑term RSI).
	•	Regime: CHOP.
	•	Exit: VWAP or fixed R multiple.
	•	Config parameters:
	•	sigma_band
	•	max_hold_minutes
	•	time_window_start, time_window_end.
	2.	Trend Pullback Scalper (BULL/BEAR)
	•	Trend detection via EMA20 vs EMA50 and regime.
	•	In BULL: buy pullbacks to EMA20 when short‑term RSI resets from overbought.
	•	In BEAR: analogous short setup.
	•	Exit: prior swing high/low or fixed R.
	•	Config parameters:
	•	ema_fast, ema_slow
	•	pullback_depth_pct
	•	entry_confirmation (RSI, candle pattern).
	3.	Opening Range Breakout/Breakdown (ORB)
	•	Define opening range (first X minutes).
	•	In BULL: go long on ORH break; BEAR: short on ORL break.
	•	Filter using:
	•	gap size
	•	premarket volume
	•	flow_zscore
	•	Config parameters:
	•	orb_minutes
	•	min_gap_pct
	•	min_flow_zscore.
	4.	Failed Breakout Fade (CHOP)
	•	Detect breakout above prior high or below prior low that quickly reverses.
	•	Enter in the opposite direction back into the range.
	•	Tight stop just beyond breakout extremes.
	5.	VWAP Trend Rider (BULL/BEAR)
	•	Only in strong trend (trend_score high).
	•	Enter when price reclaims VWAP in the direction of trend, EMAs aligned, volume uptick.
	•	Exit: trailing stop or fixed R.
	6.	Index Mean Reversion (CHOP)
	•	Only for index ETFs (SPY/QQQ).
	•	Enter when index is kσ from short‑term mean and regime=CHOP.
	•	Very small per‑trade risk.
	7.	Flow‑Confirmed Momentum (BULL/BEAR)
	•	Options flow from Unusual Whales strongly biased (flow_zscore + call_put_ratio).
	•	Enter in the direction of flow on intraday momentum breaks.
	8.	Gap‑Fill Scalper (CHOP / weak trend)
	•	Morning gaps of size X–Y%.
	•	Fade gap if extension fails early in session.
	9.	Momentum Continuation (BULL/BEAR) ✅ IMPLEMENTED
	•	Trend continuation strategy with RSI and EMA confirmation.
	•	Enter on pullback to EMA when trend is strongly aligned.
	•	Requires ADX > threshold and RSI reset from overbought/oversold.
	•	Config parameters:
	•	ema_fast, ema_slow
	•	adx_threshold
	•	rsi_entry_zone.
	10.	VIX Spike Fade (VOLATILITY) ✅ IMPLEMENTED
	•	Mean reversion on volatility spikes using VXX/VIX.
	•	Enter SHORT on extreme VIX readings (> 2σ above mean).
	•	Volatility clustering: requires confirmation of spike exhaustion.
	•	Config parameters:
	•	vix_z_threshold
	•	max_hold_minutes
	•	exit_z_threshold.

Each strategy:
	•	Has a scanner profile to identify symbols likely to fit it.
	•	Has parameters in strategies.yaml.
	•	Logs key features in Signal.meta at entry.

⸻

8. Analytics Layer

8.1 Database Schema (Conceptual)

Use SQLite for local dev, Postgres for production.

Table: trades
	•	id (PK)
	•	symbol
	•	strategy
	•	regime_at_entry
	•	regime_at_exit
	•	side
	•	qty
	•	entry_time
	•	exit_time
	•	entry_price
	•	exit_price
	•	commission
	•	slippage_estimate
	•	pnl_gross
	•	pnl_net
	•	initial_risk
	•	pnl_r
	•	mae_r
	•	mfe_r
	•	holding_period_seconds
	•	features_json (snapshot at entry)
	•	correlation_id

Table: signals
	•	id (PK)
	•	correlation_id
	•	symbol
	•	strategy
	•	regime
	•	time
	•	raw_side
	•	raw_size
	•	accepted (bool)
	•	rejection_reason (nullable)
	•	meta_json

Table: orders
	•	id (PK)
	•	correlation_id
	•	trade_id (nullable until known)
	•	symbol
	•	side
	•	qty
	•	type
	•	limit_price
	•	status
	•	time_placed
	•	time_last_update
	•	broker_order_id
	•	meta_json

Table: fills
	•	id (PK)
	•	order_id
	•	fill_time
	•	fill_price
	•	fill_qty

Table: regime_history
	•	id (PK)
	•	timestamp
	•	regime
	•	index_symbol
	•	index_price
	•	cum_ret
	•	trend_score
	•	vol

Table: scanner_snapshots
	•	id (PK)
	•	timestamp
	•	regime
	•	symbol
	•	scanner_score
	•	strategies_json
	•	features_json

Table: strategy_stats_daily
	•	date
	•	strategy
	•	regime
	•	n_trades
	•	winrate
	•	avg_r
	•	median_r
	•	std_r
	•	max_drawdown_r
	•	max_consecutive_losers
	•	pnl_r_total

Table: agent_actions
	•	id (PK)
	•	timestamp
	•	action_type (e.g. DISABLE_STRATEGY, TUNE_PARAM)
	•	strategy
	•	regime (nullable)
	•	details_json (before/after values, metrics)
	•	human_reviewed (bool)
	•	approved (bool)

8.2 Analytics Jobs

Nightly job (could also run intra‑day):
	•	Aggregate trades into strategy_stats_daily.
	•	Generate:
	•	Per‑strategy, per‑regime metrics.
	•	PnL by hour of day.
	•	Scanner score vs PnL deciles.
	•	Save HTML/Markdown summary for human review.

⸻

9. Agent Layer

9.1 Stage 1 – Strategy Health & Risk Adjustments

Input:
	•	Recent strategy_stats_daily (e.g., last 20–30 trading days).
	•	Config with thresholds:
	•	min_trades
	•	z_high for significance
	•	max allowed drawdown.

Process:

For each (strategy, regime) pair:
	1.	Filter to last N days, compute:
	•	n_trades, expectancy, std_r, winrate,
max_drawdown_r.
	2.	If n_trades < min_trades: mark as “insufficient data”.
	3.	Else compute:
	•	se = std_r / sqrt(n_trades)
	•	z = expectancy / se
	4.	Decision rules (deterministic):
	•	If z < -z_high and drawdown large:
	•	Decrease max_risk_per_trade (down to zero if needed).
	•	Possibly set enabled: false for that regime.
	•	If z > +z_high and drawdown acceptable:
	•	Optionally allow a controlled increase in max_risk_per_trade (only if flagged as allowed AND with human review by default in v1).

Output:
	•	Update strategies.auto.yaml with overrides:
	•	enabled
	•	max_risk_per_trade
	•	Record decisions in agent_actions.

9.2 Stage 2 – Parameter Tuning

For strategies with tunable params:
	•	Define param search space (e.g., VWAP band: {1.0, 1.5, 2.0}).
	•	Use either:
	•	Historical bar data for offline backtesting; or
	•	Recorded signals to simulate “subset conditions” (e.g., only signals with band < X).
	•	Compute metrics for each candidate parameter set.
	•	Choose best candidate that:
	•	Improves expectancy.
	•	Controls drawdown.
	•	Has sufficient sample size.

Agent writes:

VWAPReversion:
  params:
    band_sigma: 1.5
  metadata:
    last_optimized: "2025-12-01"
    window_days: 30

Stored in overrides file and logged to agent_actions.

9.3 Stage 3 – Code‑Level Proposals

High‑level design:
	1.	Identify problematic strategies:
	•	Persistent negative expectancy.
	•	High drawdown.
	•	Weak dependence of performance on key features.
	2.	Generate proposals:
	•	Additional filters.
	•	Alternate exit logic.
	•	New strategy variants (e.g., _v2).
	3.	Draft code in a sandbox:
	•	New strategy classes in strategies/.
	•	New or modified scanner profiles.
	4.	Backtest candidate versions:
	•	Compare with baseline.
	5.	Gate:
	•	Only if candidate passes objective thresholds.
	6.	Human approval:
	•	Agent outputs a patch/diff and summary (artifacts/proposals).
	•	Human developer reviews and merges manually.
	•	(Optional) Automated application via explicit environment variable opt-in (e.g., CERBERUS_STAGE3_APPROVED=1).

Constraints:
	•	Stage 3 must not change live trading behavior without explicit human approval (or explicit configuration opt-in).
	•	Stages 1 & 2 can change:
	•	Risk downward.
	•	Disable strategies.
	•	Tighten parameters.
without requiring human approval (config‑only changes).

⸻

10. Vertical Slice Architecture

The system must be implementable and testable incrementally via vertical slices, each forming an end‑to‑end, working pipeline.

10.1 Vertical Slice 1 – Regime‑Only Skeleton
	•	Implement:
	•	Alpaca WebSocket client for SPY bars.
	•	RegimeDetector + logging to regime_history.
	•	Minimal MarketState.
	•	Goal: verify stable, deterministic regime classification and logging.

10.2 Vertical Slice 2 – Single Symbol, Single Strategy, No Scanner
	•	Hard‑coded symbol (e.g., AAPL).
	•	Implement:
	•	SymbolState
	•	ExecutionEngine skeleton
	•	StrategyEngine with one simple strategy (e.g., VWAP Reversion with fixed params).
	•	RiskManager basic version (fixed small size, daily PnL limit).
	•	OrderExecutor stub / paper trading mode.
	•	Logging for signals, orders, trades.
	•	Goal: place simulated orders in response to strategy signals with full logging.

10.3 Vertical Slice 3 – Scanner → Execution → Analytics Loop
	•	Implement:
	•	Scanner with:
	•	UniverseBuilder
	•	FeaturePipeline (price/volume only at first)
	•	StrategyScannerProfile for VWAP Reversion
	•	Use scanner output to:
	•	Update watchlist.
	•	Set allowed_strategies per symbol.
	•	Extend analytics to:
	•	Write full rows to trades, signals, scanner_snapshots.
	•	Still use one strategy only.
	•	Goal: end‑to‑end pipeline where scanner selects symbols, engine trades, analytics records data.

10.4 Vertical Slice 4 – Multiple Strategies & Regimes
	•	Add more strategies:
	•	ORB, Trend Pullback, etc.
	•	Add multiple StrategyScannerProfiles.
	•	Enable regime‑based routing in StrategyEngine.
	•	Extend RiskManager for per‑strategy caps.
	•	Goal: multiple strategies trading multiple symbols with regime awareness.

10.5 Vertical Slice 5 – Agent Stage 1
	•	Implement:
	•	Aggregation job to populate strategy_stats_daily.
	•	Agent Stage 1:
	•	Reads stats.
	•	Writes strategies.auto.yaml.
	•	Logs agent_actions.
	•	Engine on startup:
	•	Merge base strategies.yaml with strategies.auto.yaml.
	•	Goal: strategy disabling / risk tightening based on performance.

10.6 Vertical Slice 6 – Flow Data & Advanced Analytics
	•	Integrate Unusual Whales in FeaturePipeline.
	•	Extend SymbolFeatures and select strategies that use flow.
	•	Add more advanced analytics:
	•	PnL vs scanner score deciles.
	•	Flow‑conditioned performance.
	•	Future slices:
	•	Stage 2 & 3 Agent logic.

Each vertical slice must:
	•	Compile and run as a coherent app.
	•	Include meaningful logging and error handling.
	•	Have tests around core logic (regime classification, strategy behavior, risk capping).

⸻

11. Nonfunctional Requirements

11.1 Determinism
	•	No use of non‑seeded randomness in production logic.
	•	Given the same:
	•	Historical data
	•	Config
	•	Start time
	•	The system must produce identical:
	•	Signals
	•	Orders
	•	Trades

11.2 Performance
	•	Able to handle:
	•	Up to 30 tickers with 1‑minute bars (optionally faster bars).
	•	Latency:
	•	On each bar, end‑to‑end processing from bar receipt to order submission should be comfortably under a second in Python.

11.3 Robust Error Logging & Observability
	•	Use structured logging (e.g., JSON) with:
	•	timestamp
	•	level
	•	module
	•	correlation_id where applicable
	•	symbol, strategy, regime for domain errors
	•	Log levels:
	•	DEBUG – development & detailed tracing.
	•	INFO – normal lifecycle events (startup, scan results, regime changes).
	•	WARN – recoverable issues (missing data, API timeouts with fallback).
	•	ERROR – unexpected exceptions, order failures.
	•	For each major module:
	•	Scanner: log number of symbols processed, number filtered, failures per data source.
	•	RegimeDetector: log regime changes and any computation errors.
	•	ExecutionEngine:
	•	log per‑bar processing exceptions.
	•	log strategy exceptions individually.
	•	RiskManager:
	•	log signal rejections with reasons.
	•	OrderExecutor:
	•	log order submissions and failures with broker response payloads.
	•	Include minimal health metrics:
	•	Number of bars processed.
	•	Number of signals generated.
	•	Number of orders submitted.
	•	Error counts per module.

11.4 Fault Tolerance & Degradation
	•	If Unusual Whales API fails:
	•	Log at WARN/ERROR.
	•	Set flow features to neutral values.
	•	Continue running with reduced feature set.
	•	If Alpaca WebSocket temporarily disconnects:
	•	Attempt reconnection with backoff.
	•	On reconnection, ensure state is consistent (e.g., reload positions from REST).
	•	If DB is temporarily unavailable:
	•	Log error.
	•	Use in‑memory queue/buffer and retry (bounded).
	•	If persistent failure, system can choose to:
	•	Halt trading safely, or
	•	Continue trading with explicit warning (configurable).

⸻

12. Advanced Exit System ✅ IMPLEMENTED

This section documents the dynamic exit management system that replaces static broker-managed bracket orders with self-managed exits for greater control and parity.

12.1 Architectural Shift
	•	Broker-managed exits disabled when advanced_exits.enabled = true.
	•	PositionManager becomes the single source of truth for all exit decisions.
	•	Ensures 100% parity between live trading and backtesting.

12.2 Trailing Stops
	•	Configuration: TrailingStopConfig (enabled, trail_pct, min_profit_to_activate).
	•	High-water mark tracking: Position.trailing_high_water.
	•	Ratchet mechanism: Stop only moves in favorable direction.
	•	Implementation: PositionManager._update_trailing_stop().

12.3 Partial Profit Taking
	•	Configuration: PartialExitConfig (first_exit_at_r, first_exit_pct).
	•	Trigger: 1R profit with 50% scale-out (configurable).
	•	State tracking: Position.partial_exits_taken prevents re-triggering.
	•	Implementation: PositionManager._check_partial_exit().

12.4 Regime-Aware Stop Multipliers
	•	Volatility-based stop width adjustment.
	•	Multipliers: { low: 0.75, normal: 1.0, high: 1.5, shock: 2.0 }.
	•	Captured at entry via Position.regime_stop_multiplier.

12.5 Configuration Model

class AdvancedExitsConfig(BaseModel):
    enabled: bool = False
    trailing_stop: TrailingStopConfig
    partial_exits: PartialExitConfig
    regime_aware_stops: bool = True
    regime_stop_multipliers: Dict[str, float]

⸻

13. Backtesting Engine ✅ IMPLEMENTED

This section documents the backtesting system that achieves 100% logic parity with live trading.

13.1 Architecture
	•	BacktestRunner instantiates the production ExecutionEngine.
	•	BacktestOrderExecutor provides mock execution with configurable realism.
	•	BacktestFeaturePipeline computes features from historical data.

13.2 Execution Realism

Volume-Aware Partial Fills:
	•	partial_fill_mode: none | fixed | volume_aware
	•	Calculates fill_qty = min(order_qty, bar.volume * partial_fill_rate).
	•	Prevents infinite liquidity assumption.

Volume-Impact Slippage:
	•	slippage_mode: fixed | volume_impact
	•	Formula: base_bps * (1.0 + (qty / volume) * impact_mult).

ATR-Based Spread:
	•	spread_mode: fixed | atr_based
	•	Scales bid-ask spread with ATR-derived volatility.

Bracket Exit Modes:
	•	stop_first (default): Conservative; stop wins on same-bar triggers.
	•	best_exit: Target wins on same-bar triggers (matches production M3).

13.3 Gap Awareness
	•	Gaps past exit levels fill at bar open price.
	•	Realistic slippage for stop events, better fills for gap-up targets.

13.4 Daily Equity Reset
	•	Optional: daily_equity_reset: true.
	•	Resets cash to initial value ($100k) each session.
	•	Prevents compounding drawdown from biasing research trials.

13.5 Flow Strategy Gating
	•	disable_flow_strategies: true skips flow-dependent strategies.
	•	Flow features are neutral (zeros) in backtests without offline flow source.

13.6 Configuration

backtest:
  partial_fill_mode: volume_aware
  partial_fill_rate: 0.10
  slippage_mode: volume_impact
  slippage_bps: 5.0
  slippage_impact_mult: 2.0
  spread_mode: atr_based
  bracket_exit_mode: best_exit
  daily_equity_reset: true
  disable_flow_strategies: true

⸻

# PRD Regime Upgrade Patch (v2) ✅ IMPLEMENTED

This patch upgrades the PRD “Multi‑Strategy Intraday Scalping System (Equities, Alpaca + Unusual Whales)” from a single **SPY‑only BULL/BEAR/CHOP** label to a **multi‑axis market context + per‑symbol micro‑context** regime system.

It keeps the original design goals:
- vertical-slice architecture
- deterministic behavior
- robust, structured logging
- agent-driven config improvements

---

## 0) Why change the current regime approach?

**Current behavior in the PRD**
- One global regime label derived from SPY intraday bars:
  - features: `cum_ret`, `trend_score = abs(cum_ret)/vol`
  - label: `BULL / BEAR / CHOP`
  - smoothing: majority vote over last K
- Strategies are routed via `strategies_by_regime` in `StrategyEngine`.

**Main issues**
1. **A single label collapses distinct worlds**
   “CHOP” can mean low-vol driftless noise or high-vol whipsaw. Those are opposite for sizing and execution risk.

2. **Global SPY regime ≠ symbol regime**
   AAPL can be trending hard on news while SPY chops.

3. **Hard gating causes missed opportunity and mode errors**
   ORB and momentum setups can exist even in a “CHOP” market label; conversely mean-reversion gets murdered during volatility shocks.

4. **No confidence / uncertainty handling**
   Early session classifications are unstable; the system should express “I’m not sure yet” and size down accordingly.

---

## 1) PRD edits: new “Regime” concept

### 1.1 Replace “Market Regime Detector” with “Market Context & Regime Service”

Rename component **3. Market Regime Detector** → **3. Market Context & Regime Service**.

**Outputs**
- A *vector* of discrete states (“axes”) + continuous features
- A confidence score per axis
- A compact derived label for humans (optional), e.g. `BULL/BEAR/CHOP` remains as a *display* tag only

---

## 2) Domain model changes

### 2.1 Replace `Regime` enum

Remove:
```python
class Regime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    CHOP = "chop"
```

Add axis enums:

```python
from enum import Enum

class TrendRegime(str, Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"

class VolRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    SHOCK = "shock"      # “stop trading / extreme caution” zone

class LiquidityRegime(str, Enum):
    GOOD = "good"
    THIN = "thin"
    STRESSED = "stressed"  # spreads/impact unacceptable

class RiskRegime(str, Enum):
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"

class SessionRegime(str, Enum):
    PREMARKET = "premarket"
    OPENING = "opening"
    MIDDAY = "midday"
    POWER_HOUR = "power_hour"
    CLOSE = "close"
```

### 2.2 Add `MarketRegimeSnapshot`

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

@dataclass(frozen=True)
class MarketRegimeSnapshot:
    time: datetime

    # key symbols used
    index_symbol: str             # e.g. "SPY"
    vol_symbol: str | None        # e.g. "VXX" (optional but recommended)

    # discrete regimes (axes)
    trend: TrendRegime
    vol: VolRegime
    liquidity: LiquidityRegime
    risk: RiskRegime
    session: SessionRegime

    # continuous features (always logged)
    cum_ret: float
    trend_strength: float
    realized_vol: float
    vol_of_vol: float
    liquidity_score: float
    risk_score: float

    # deterministic uncertainty
    confidence: Dict[str, float]  # per axis, 0..1

    # reproducibility
    model_version: str            # bump when logic/thresholds change
```

### 2.3 Update `MarketState`

Replace:
```python
@dataclass
class MarketState:
    time: datetime
    regime: Regime
    index_symbol: str
    index_price: float
    index_return: float
    realized_vol: float
    ...
```

With:
```python
@dataclass
class MarketState:
    time: datetime
    regime: MarketRegimeSnapshot

    index_price: float
    index_return: float

    daily_pnl: float
    risk_mode: RiskMode
    meta: Dict[str, Any]
```

### 2.4 Update `Signal` and logging payloads

Replace `Signal.regime: Regime` with a compact tag map:

```python
@dataclass
class Signal:
    ...
    regime_tags: Dict[str, str]     # {"trend":"up","vol":"high","risk":"risk_off",...}
    regime_confidence: Dict[str, float]
    ...
```

You do **not** need to carry the full MarketRegimeSnapshot into every object at runtime; you *do* need to log enough to reproduce and analyze.

---

## 3) Market Context & Regime Service: v1 logic (deterministic)

### 3.1 Inputs

Always-on subscriptions (count toward the 30 ticker WebSocket limit):
- SPY (index)
- **One** volatility proxy ETF if available in your data feed: VXX (preferred) or similar
- Optional but helpful: QQQ, IWM (broad risk proxy triangulation)

Session/time inputs:
- exchange timezone: America/New_York
- market open/close times

### 3.2 Features

Keep your existing `cum_ret` and `trend_score`, and add:

- **vol baseline**: rolling median vol over last M windows (intraday adaptive)
- **vol-of-vol**: rolling std of realized_vol
- **shock flag**:
  - |1-min return| > k * rolling_vol
  - OR 1-min range% > threshold
- **liquidity proxy** (bars-only fallback):
  - dollar_volume = close * volume
  - range_pct = (high - low)/close
  - liquidity_score = dollar_volume / (range_pct + eps)
  - If you have quotes: use effective spread / NBBO spread directly (better).
- **risk score** (simple v1):
  - risk_score = a*(SPY return) - b*(VXX return)
  - if VXX unavailable: use SPY return + SPY realized_vol increase

### 3.3 Classification per axis

**Trend**
- if `trend_strength < trend_flat_thresh` → FLAT
- else sign(cum_ret) → UP/DOWN

**Vol**
- compute z = realized_vol / (baseline_vol + eps)
- z < low_thresh → LOW
- low_thresh <= z < high_thresh → NORMAL
- high_thresh <= z < shock_thresh → HIGH
- z >= shock_thresh OR shock_flag → SHOCK

**Liquidity**
- liquidity_score quantiles:
  - top quantile → GOOD
  - middle → THIN
  - bottom + shock_flag → STRESSED

**Risk**
- risk_score thresholds:
  - above +t → RISK_ON
  - below -t → RISK_OFF
  - else NEUTRAL

**Session**
- derived from wall clock:
  - opening = first 30–60 min
  - midday = lunch hours
  - power_hour = last 60 min

### 3.4 Smoothing / hysteresis (per axis, not one global vote)

Replace the single majority vote with:
- per-axis hysteresis
- minimum hold time (e.g., do not switch vol regime more than once per X minutes unless SHOCK)

This reduces “regime flapping” in exactly the environments that trigger overtrading.

---

## 4) Strategy selection changes (most important PRD edit)

### 4.1 Remove `strategies_by_regime` routing

Current PRD uses:
```python
regime_strats = set(self.strategies_by_regime.get(regime, []))
active = allowed ∩ regime_strats
```

Replace with a deterministic **StrategyActivationPolicy** that evaluates rules:

```python
@dataclass(frozen=True)
class StrategyActivationPolicy:
    # allowed lists; empty => “no constraint”
    session: list[SessionRegime]
    trend: list[TrendRegime]
    vol: list[VolRegime]
    liquidity: list[LiquidityRegime]
    risk: list[RiskRegime]

    # optional numeric constraints
    min_confidence: float = 0.0     # require regime confidence
```

Each strategy has an activation policy in config, and the StrategyEngine does:

1. check symbol is allowed by scanner
2. check market regime axes match activation policy
3. check any strategy-specific prerequisites (flow availability, ATR, spread, etc.)
4. run strategy

### 4.2 Configuration example (strategies.yaml)

```yaml
VWAPReversion:
  enabled: true
  activation:
    session: [opening, midday, power_hour]
    trend: [flat]
    vol: [low, normal]
    liquidity: [good, thin]
    risk: [risk_on, neutral]
    min_confidence: 0.60

ORB:
  enabled: true
  activation:
    session: [opening]
    vol: [normal, high]
    liquidity: [good]
    min_confidence: 0.40
  requirements:
    flow_required: false   # set true if you want ORB only when flow is present
```

This is strictly more expressive than BULL/BEAR/CHOP without becoming a scientific paper.

---

## 5) Risk management changes

### 5.1 Add regime-based risk scaling

Add to `risk.yaml`:

```yaml
regime_risk_multipliers:
  vol:
    low: 1.10
    normal: 1.00
    high: 0.60
    shock: 0.00
  liquidity:
    good: 1.00
    thin: 0.75
    stressed: 0.00
  risk:
    risk_on: 1.00
    neutral: 0.85
    risk_off: 0.50
```

RiskManager sizing:
- base_qty determined by entry→stop risk
- final_qty = base_qty * multipliers[vol] * multipliers[liquidity] * multipliers[risk]
- enforce min trade size; otherwise reject signal with reason “REGIME_RISK_ZERO” or “REGIME_RISK_BELOW_MIN”.

### 5.2 Automatic `risk_mode`

In MarketState update loop:
- if vol == SHOCK or liquidity == STRESSED → set `risk_mode = OFF`
- elif vol == HIGH or risk == RISK_OFF → `risk_mode = REDUCED`
- else `risk_mode = NORMAL`

Allow manual override (config) but log overrides loudly.

---

## 6) Scanner changes

Scanner already computes symbol features (ADX, zscore, vwap distance, etc.).
Change the scanner’s dependence on a single regime label:

- Pass the full `MarketRegimeSnapshot` (or at least `regime_tags`) to `StrategyScannerProfile.score()`.
- Add a baseline liquidity/spread filter (critical for scalping) so you don’t nominate symbols that are untradeable under current conditions.
- If flow features are missing, **do not set to neutral** for strategies where flow is a prerequisite. Instead:
  - record `feature_availability.flow = false`
  - allow `min_requirements()` to reject the symbol for flow-dependent strategies

---

## 7) Analytics schema changes

### 7.1 Regime history table

Replace:
- timestamp, regime, cum_ret, trend_score, vol

With:

- timestamp
- model_version
- trend, vol, liquidity, risk, session
- cum_ret, trend_strength, realized_vol, vol_of_vol
- liquidity_score, risk_score
- confidence_json

### 7.2 Trades/signals tables

Replace `regime_at_entry` and `regime_at_exit` with:
- `regime_tags_entry_json`
- `regime_tags_exit_json`
- optional “compact label” column for convenience: `regime_compact_entry` (e.g., “UP/HIGH/RISK_OFF”)

---

## 8) Agent changes

The combinatorial trap: if you do full cartesian products of regimes, you get sparse bins and nonsense.

So adjust Stage 1:

- Compute stats **per axis**:
  - performance by vol regime
  - performance by trend regime
  - performance by liquidity regime
  - performance by session regime
- Then compute **one** or **two** high-impact joint bins only (configurable), e.g.:
  - vol × session
  - vol × trend

Agent outputs:
- disable strategies in specific axis states (e.g., “VWAPReversion disabled when vol=HIGH”)
- adjust risk multipliers downward when a regime becomes toxic

All changes remain config-only.

---

## 9) Vertical slice updates

Update slice plan:

### Slice 1 – Market Context Skeleton
- Subscribe to SPY (+VXX if used)
- Compute MarketRegimeSnapshot
- Log to `regime_history` with model_version
- Unit tests for each axis classification + hysteresis

### Slice 2 – Risk scaling by regime
- No strategy changes yet
- RiskManager applies multipliers + logs sizing inputs
- Paper-trade one strategy to verify drawdown reduction in high-vol/shock regimes

### Slice 3 – Strategy activation rules
- Replace `strategies_by_regime` routing with ActivationPolicy evaluation
- Add tests ensuring deterministic activation across bars

---

## 10) Logging & error taxonomy additions

Add structured fields to all logs touching regimes:

- `regime.model_version`
- `regime.tags` (trend/vol/liquidity/risk/session)
- `regime.confidence`
- `regime.features` (cum_ret, realized_vol, liquidity_score, risk_score)

Add explicit error codes:
- `REGIME_COMPUTE_FAILED`
- `REGIME_DATA_MISSING`
- `REGIME_RISK_ZERO`
- `STRATEGY_ACTIVATION_BLOCKED`

---

## 11) Minimal viable set of changes (if you want it lean)

If you want the smallest change with maximum benefit:
1. Add **vol regime** (LOW/NORMAL/HIGH/SHOCK) + risk multipliers
2. Add **session regime** (OPENING/MIDDAY/POWER_HOUR)
3. Replace hard regime gating with ActivationPolicy (even if it only checks vol+session)

This already fixes most “simple BULL/BEAR/CHOP” failure modes for scalping.
