import math
import time as _time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session

from src.analysis.db import DatabaseDatabase
from src.analysis.regime import Regime
from src.analysis.schema import Fill as DbFill
from src.analysis.schema import Order as DbOrder
from src.analysis.schema import ScannerSnapshot
from src.analysis.schema import Signal as DbSignal
from src.analysis.schema import Trade as DbTrade
from src.analytics.meta_labeler import MetaLabeler
from src.core.domain import (
    Position,
    ScanResult,
    Side,
    WatchlistSymbol,
)
from src.core.errors import ErrorCode
from src.core.indicators import RollingEMA, RollingRSI, RollingSMA, RollingStd
from src.core.logger import StructuredLogger
from src.engine.health import HealthMonitor
from src.engine.market import MarketStateManager
from src.engine.position_manager import PositionManager
from src.engine.risk import RiskManager
from src.engine.strategy_engine import StrategyEngine, StrategyRouting
from src.scanner.core import Scanner
from src.scanner.streaming_scanner import StreamingScanner
from src.strategies.base import BaseStrategy, Signal, SymbolState
from src.strategies.config_models import build_activation_policies_from_config


class ExecutionEngine:
    """
    Orchestrates data flow, strategy execution, and order management.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: StructuredLogger,
        db: Optional[DatabaseDatabase] = None,
        run_id: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.config = config
        self.logger = logger
        self.db = db
        self.run_id = run_id
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))

        self.risk_manager = RiskManager(config, logger)
        self.order_executor = None  # Set externally via main.py
        # Optional broker client (e.g. AlpacaClient) for direct broker operations
        # like flatten_all and reconciliation. Set externally via main.py if needed.
        self.broker_client: Any = None
        # Callback for subscription changes: (action: "add"|"remove", symbol: str) -> None
        self.on_subscription_change: Optional[Callable[[str, str], None]] = None
        self.scanner: Optional[Scanner] = None  # Injected later or via init
        self.streaming_scanner: Optional[StreamingScanner] = None  # Injected later

        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_engine: Optional[StrategyEngine] = None

        # Wire CerberusLedgerAdapter into PositionManager (best-effort)
        ledger_adapter = None
        try:
            from empire_core.ledger import LedgerWriter

            from src.core.ledger_adapter import CerberusLedgerAdapter

            ledger_db_path = config.get("ledger_db_path", "ledger.db")
            writer = LedgerWriter(db_path=ledger_db_path, system="cerberus")
            ledger_adapter = CerberusLedgerAdapter(ledger=writer)
            logger.info("ledger_adapter_attached", ledger_db=ledger_db_path)
        except Exception:
            logger.warning("ledger_adapter_init_failed", exc_info=True)

        self.position_manager = PositionManager(ledger_adapter=ledger_adapter)
        self.meta_labeler = MetaLabeler(logger)
        self.symbol_states: Dict[str, SymbolState] = {}
        # Bounded trade capture for backtests/analysis
        self.closed_trades: deque = deque(maxlen=5000)
        # Fill dedup: prevent double processing of the same fill on reconnect/restart.
        # Pre-populated from DB on startup to survive restarts (Finding 4, consensus 4/8).
        self._processed_fill_ids: Set[str] = set()
        self._restore_fill_dedup_from_db()

        # Backtest mode flag — set by runner.py to skip DB writes, health monitoring, etc.
        self.backtest_mode: bool = False
        # Cached indicator periods (computed once, reused per bar)
        self._cached_indicator_periods: Optional[Dict[str, set]] = None

        # Extracted Collaborators
        self.health = HealthMonitor(config, logger, run_id, self.clock)
        self._last_regime: Optional[Regime] = None  # M1 fix: Track regime changes
        self._last_cache_clear_date: Optional[str] = None  # M1 fix: Track daily clears
        self._reconcile_interval_sec = 0  # set from config.health
        self.market_manager = MarketStateManager(config, logger, db, self.clock, on_error=self._inc_error)

        # For backward compatibility (tests accessing engine.market_state)
        # We wrap the manager's state property.
        # Ideally tests should update to use self.market_manager.state, but property is safer for refactor.

        # Throttling config
        self.max_churn_per_scan = config.get("max_churn_per_scan", 2)

        # Fail-fast Config
        self.consecutive_on_bar_errors = 0
        self.max_consecutive_errors = int(config.get("max_consecutive_errors", 5))

        # PRD 3.3: keep MarketState.risk_mode aligned with RiskManager at startup.
        self._set_risk_mode(self.risk_manager.risk_mode)

        self.account = None  # Holds Alpaca Account object

    @property
    def market_state(self):
        return self.market_manager.state

    @market_state.setter
    def market_state(self, value):
        self.market_manager.state = value

    @property
    def bars_processed(self):
        return self.health.bars_processed

    @bars_processed.setter
    def bars_processed(self, value):
        self.health.bars_processed = value

    @property
    def signals_generated(self):
        return self.health.signals_generated

    @signals_generated.setter
    def signals_generated(self, value):
        self.health.signals_generated = value

    @property
    def orders_submitted(self):
        return self.health.orders_submitted

    @orders_submitted.setter
    def orders_submitted(self, value):
        self.health.orders_submitted = value

    @property
    def error_counts(self):
        return self.health.error_counts

    @error_counts.setter
    def error_counts(self, value):
        self.health.error_counts = value

    def _inc_error(self, module: str) -> None:
        self.health.record_error(module)

    def _set_risk_mode(self, mode: str) -> None:
        """
        PRD 3.3 / 6.5: keep RiskManager + MarketState risk_mode synchronized.
        """
        m = str(mode or "normal").lower()
        try:
            self.risk_manager.risk_mode = m
        except Exception as e:
            self.logger.debug(
                "Bar processing metrics update failed",
                operation="update_bar_metrics",
                error=str(e),
            )

        self.market_manager.set_risk_mode(m)

    def _sanitize_features_snapshot(self, value: Any) -> Any:
        """
        Best-effort conversion to keep feature snapshots JSON-serializable (PRD 8.1).

        Currently, SymbolFeatures contains `datetime` fields (e.g., last_updated) which should
        be stored as strings when persisting in JSON columns.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: self._sanitize_features_snapshot(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_features_snapshot(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._sanitize_features_snapshot(v) for v in value)
        return value

    def _update_indicator_cache(self, state: SymbolState, bar: Any) -> None:
        """
        Hot-path indicator caching to avoid per-strategy pandas work (PRD 11.2).

        Stores scalar values under deterministic keys:
        - ema_close:{period}, ema_close:{period}:prev
        - rsi:{period}, rsi:{period}:prev
        - sma_vol:{period}, sma_vol:{period}:prev
        - bb_mean:{period}, bb_std:{period}
        """
        try:
            if not isinstance(state.indicators, dict):
                return

            periods = self._collect_indicator_periods(state)
            close = float(getattr(bar, "close", 0.0) or 0.0)
            volume = float(getattr(bar, "volume", 0.0) or 0.0)

            self._update_ema_indicators(state, close, periods["ema"])
            self._update_rsi_indicators(state, close, periods["rsi"])
            self._update_vol_sma_indicators(state, volume, periods["vol_sma"])
            self._update_bb_indicators(state, close, periods["bb"])
        except Exception as e:
            self._inc_error("execution")
            self.logger.error(
                "Indicator cache update failed — strategies may use stale values",
                symbol=getattr(state, "symbol", "unknown"),
                error=str(e),
                exc_info=True,
            )

    def _collect_indicator_periods(self, state: SymbolState) -> Dict[str, set]:
        """Collect required indicator periods from enabled strategies.

        Results are cached on self._cached_indicator_periods after the first call
        to avoid re-iterating strategy configs on every bar.
        """
        if self._cached_indicator_periods is not None:
            return self._cached_indicator_periods

        strategies = list(getattr(state, "allowed_strategies", []) or [])
        strat_cfgs = self.config.get("strategies") if isinstance(self.config.get("strategies"), dict) else {}

        periods: Dict[str, set[int]] = {
            "ema": set(),
            "rsi": set(),
            "vol_sma": set(),
            "bb": set(),
        }

        for s in strategies:
            cfg = strat_cfgs.get(s) if isinstance(strat_cfgs, dict) else None
            if not isinstance(cfg, dict):
                cfg = {}
            if s in ("trend_pullback", "vwap_trend_rider"):
                periods["ema"].add(int(cfg.get("ema_fast", 20)))
                periods["ema"].add(int(cfg.get("ema_slow", 50)))
            if s == "trend_pullback":
                periods["rsi"].add(int(cfg.get("rsi_len", 2)))
            if s in ("flow_momentum", "vwap_trend_rider"):
                periods["vol_sma"].add(20)
            if s == "index_mean_reversion":
                periods["bb"].add(int(cfg.get("bb_len", 20)))

        self._cached_indicator_periods = periods
        return periods

    def _update_ema_indicators(self, state: SymbolState, close: float, periods: set) -> None:
        """Update EMA indicators for given periods."""
        for p in sorted(x for x in periods if x > 0):
            key = f"_ema_close_obj:{p}"
            obj = state.indicators.get(key)
            if not isinstance(obj, RollingEMA):
                obj = RollingEMA.from_period(p)
                state.indicators[key] = obj
            prev = obj.value
            val = obj.update(close)
            state.indicators[f"ema_close:{p}:prev"] = prev
            state.indicators[f"ema_close:{p}"] = val

    def _update_rsi_indicators(self, state: SymbolState, close: float, periods: set) -> None:
        """Update RSI indicators for given periods."""
        for p in sorted(x for x in periods if x > 0):
            key = f"_rsi_obj:{p}"
            obj = state.indicators.get(key)
            if not isinstance(obj, RollingRSI):
                obj = RollingRSI(period=p)
                state.indicators[key] = obj
            prev = obj.value
            rsi_val = obj.update(close)
            state.indicators[f"rsi:{p}:prev"] = prev
            state.indicators[f"rsi:{p}"] = rsi_val

    def _update_vol_sma_indicators(self, state: SymbolState, volume: float, periods: set) -> None:
        """Update volume SMA indicators for given periods."""
        for p in sorted(x for x in periods if x > 0):
            key = f"_sma_vol_obj:{p}"
            obj = state.indicators.get(key)
            if not isinstance(obj, RollingSMA):
                obj = RollingSMA.create(p)
                state.indicators[key] = obj
            prev = obj.value
            val = obj.update(volume)
            state.indicators[f"sma_vol:{p}:prev"] = prev
            state.indicators[f"sma_vol:{p}"] = val

    def _update_bb_indicators(self, state: SymbolState, close: float, periods: set) -> None:
        """Update Bollinger Band indicators for given periods."""
        for p in sorted(x for x in periods if x > 0):
            key = f"_std_close_obj:{p}"
            obj = state.indicators.get(key)
            if not isinstance(obj, RollingStd):
                obj = RollingStd.create(p)
                state.indicators[key] = obj
            mean, std = obj.update(close)
            state.indicators[f"bb_mean:{p}"] = mean
            state.indicators[f"bb_std:{p}"] = std

    def flatten_all(self, *, reason: str = "eod") -> None:
        """Emergency flatten: close all positions and cancel all orders."""
        if not self.broker_client:
            self.logger.warning("Flatten requested but no broker client available — skipping", reason=reason)
            # Still reset local state so the engine is in a clean state
            self._flatten_reset_local_state()
            return

        self._set_risk_mode("off")
        mismatch_mode = str(self.config.get("position_mismatch_mode", "halt")).lower()
        self.logger.info("Flatten starting", reason=reason, mismatch_mode=mismatch_mode)

        # Cancel and close
        self._flatten_cancel_orders(reason)
        self._flatten_close_positions(reason, mismatch_mode)

        # Confirm flat state
        open_positions, open_orders, confirmed = self._flatten_confirm_state()

        # Reset local state
        self._flatten_reset_local_state()

        # Handle confirmation failures
        self._flatten_handle_result(mismatch_mode, confirmed, open_positions, open_orders, reason)

    def _flatten_cancel_orders(self, reason: str) -> None:
        """Cancel Cerberus-owned open orders only (best-effort).

        The Alpaca paper account is shared across multiple Empire trading services
        (Orbit, 3Roses, Cerberus, etc.). Calling `cancel_orders()` would cancel
        EVERY open order in the account. We must scope cancellation to orders
        whose `client_order_id` begins with `cerberus_` — the prefix stamped by
        `Signal.__post_init__` in `src/core/domain.py`.
        """
        assert self.broker_client is not None
        trading_client = self.broker_client.trading_client

        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            resp = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
            open_orders = resp if isinstance(resp, list) else []
        except Exception as e:
            self.logger.error(
                "Fetch open orders failed during flatten",
                reason=reason,
                error=str(e),
                exc_info=True,
            )
            return

        cancelled = 0
        skipped = 0
        for order in open_orders:
            coid = str(getattr(order, "client_order_id", "") or "")
            if not coid.startswith("cerberus_"):
                skipped += 1
                continue
            order_id = getattr(order, "id", None)
            if order_id is None:
                continue
            try:
                trading_client.cancel_order_by_id(order_id)
                cancelled += 1
            except Exception as e:
                self.logger.error(
                    "Cancel order failed",
                    reason=reason,
                    order_id=str(order_id),
                    client_order_id=coid,
                    error=str(e),
                    exc_info=True,
                )

        self.logger.info(
            "Flatten cancel_orders scoped to Cerberus",
            reason=reason,
            cancelled=cancelled,
            skipped_foreign=skipped,
        )

    def _flatten_close_positions(self, reason: str, mismatch_mode: str) -> None:
        """Close Cerberus-owned positions only (best-effort).

        The Alpaca paper account is shared across multiple Empire trading services.
        Calling `close_all_positions()` would liquidate EVERY position in the
        account — including ones owned by Orbit, 3Roses, and any other service.

        We identify Cerberus-owned positions by matching each open position to
        its most recent filled opening order in the account. If that order's
        `client_order_id` starts with `cerberus_`, we close it; otherwise we
        leave it alone.
        """
        assert self.broker_client is not None
        trading_client = self.broker_client.trading_client

        try:
            positions = list(trading_client.get_all_positions())
        except Exception as e:
            self.logger.error(
                "Fetch positions failed during flatten",
                reason=reason,
                error=str(e),
                exc_info=True,
            )
            if mismatch_mode in ("halt", "stop", "raise"):
                raise
            return

        if not positions:
            return

        # Build symbol -> most-recent-filled-opening-order map from recent closed orders.
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            resp = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500))
            closed_orders = resp if isinstance(resp, list) else []
        except Exception as e:
            self.logger.error(
                "Fetch closed orders failed during flatten — cannot determine ownership, skipping close",
                reason=reason,
                error=str(e),
                exc_info=True,
            )
            if mismatch_mode in ("halt", "stop", "raise"):
                raise
            return

        # Map symbol -> client_order_id of the most recent filled order that could
        # have opened the current position. We prefer the most recently filled order
        # per symbol regardless of side; whoever opened the current position last
        # is the owner of record.
        latest_owner_coid: Dict[str, str] = {}
        latest_owner_ts: Dict[str, Any] = {}
        for order in closed_orders:
            try:
                status = str(getattr(order, "status", "") or "").lower()
                if status != "filled":
                    continue
                sym = str(getattr(order, "symbol", "") or "")
                if not sym:
                    continue
                coid = str(getattr(order, "client_order_id", "") or "")
                filled_at = getattr(order, "filled_at", None) or getattr(order, "updated_at", None)
                prev_ts = latest_owner_ts.get(sym)
                if prev_ts is None or (filled_at is not None and filled_at > prev_ts):
                    latest_owner_coid[sym] = coid
                    latest_owner_ts[sym] = filled_at
            except Exception:
                continue

        closed = 0
        skipped_foreign = 0
        skipped_unknown = 0
        first_exc: Optional[Exception] = None
        for pos in positions:
            sym = str(getattr(pos, "symbol", "") or "")
            if not sym:
                continue
            owner_coid = latest_owner_coid.get(sym, "")
            if not owner_coid:
                # No recent filled order found — cannot prove this is ours.
                # Err on the safe side and leave it alone.
                skipped_unknown += 1
                self.logger.warning(
                    "Skipping position with no identifiable owner order",
                    reason=reason,
                    symbol=sym,
                )
                continue
            if not owner_coid.startswith("cerberus_"):
                skipped_foreign += 1
                self.logger.info(
                    "Skipping foreign-owned position during flatten",
                    reason=reason,
                    symbol=sym,
                    owner_prefix=owner_coid.split("_", 1)[0] if "_" in owner_coid else owner_coid[:16],
                )
                continue
            try:
                trading_client.close_position(sym)
                closed += 1
            except Exception as e:
                self.logger.error(
                    "Close position failed",
                    reason=reason,
                    symbol=sym,
                    error=str(e),
                    exc_info=True,
                )
                if first_exc is None:
                    first_exc = e

        self.logger.info(
            "Flatten close_positions scoped to Cerberus",
            reason=reason,
            closed=closed,
            skipped_foreign=skipped_foreign,
            skipped_unknown=skipped_unknown,
        )

        if first_exc is not None and mismatch_mode in ("halt", "stop", "raise"):
            raise first_exc

    def _flatten_confirm_state(self) -> Tuple[list, list, bool]:
        """Confirm flat state at broker."""
        open_positions = []
        open_orders = []
        positions_confirmed = False
        orders_confirmed = False

        assert self.broker_client is not None
        try:
            open_positions = list(self.broker_client.trading_client.get_all_positions())
            positions_confirmed = True
        except Exception as e:
            self.logger.warning("Failed to fetch broker positions after flatten", error=str(e))

        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            resp = self.broker_client.trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
            open_orders = resp if isinstance(resp, list) else []
            orders_confirmed = True
        except Exception as e:
            self.logger.warning("Failed to fetch broker orders after flatten", error=str(e))

        return open_positions, open_orders, (positions_confirmed and orders_confirmed)

    def _flatten_reset_local_state(self) -> None:
        """Reset local position and order state."""
        for state in self.symbol_states.values():
            state.position = None
            state.open_orders = {}

    def _flatten_handle_result(
        self,
        mismatch_mode: str,
        confirmed: bool,
        open_positions: list,
        open_orders: list,
        reason: str,
    ) -> None:
        """Handle flatten confirmation result."""
        if mismatch_mode in ("halt", "stop", "raise") and not confirmed:
            self.logger.error(
                "Flatten confirmation failed",
                positions_confirmed=not bool(open_positions),
                orders_confirmed=not bool(open_orders),
                mismatch_mode=mismatch_mode,
            )
            raise RuntimeError("Flatten confirmation failed")

        if open_positions or open_orders:
            self.logger.error(
                "Flatten incomplete",
                open_positions=len(open_positions),
                open_orders=len(open_orders),
                mismatch_mode=mismatch_mode,
            )
            if mismatch_mode in ("halt", "stop", "raise"):
                raise RuntimeError("Flatten incomplete: broker still reports positions/orders")
        else:
            self.logger.info("Flatten complete", reason=reason)

    def _maybe_log_health(self, now: datetime) -> None:
        self.health.check_and_log(now)

    def register_strategy(self, strategy: BaseStrategy):
        """
        Registers a strategy with the engine.
        """
        self.strategies[strategy.name] = strategy
        self.logger.info("Strategy registered", strategy=strategy.name)
        self._refresh_strategy_engine()

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Applies a new merged config (e.g., after Agent Stage 1 writes overrides).

        This updates routing and risk controls deterministically without requiring a process restart.
        """
        self.config = config
        self.max_churn_per_scan = config.get("max_churn_per_scan", self.max_churn_per_scan)
        self.health.update_config(config)
        # RiskManager already has raw_config; update individual fields instead
        risk_cfg = (config.get("risk") if isinstance(config.get("risk"), dict) else None) or {}
        self.risk_manager.max_daily_loss = float(
            risk_cfg.get(
                "max_daily_loss",
                config.get("max_daily_loss", self.risk_manager.max_daily_loss),
            )
        )
        self.risk_manager.max_risk_per_trade = float(
            risk_cfg.get(
                "max_risk_per_trade",
                config.get("max_risk_per_trade", self.risk_manager.max_risk_per_trade),
            )
        )
        self.risk_manager.max_open_risk = float(
            risk_cfg.get(
                "max_open_risk",
                config.get("max_open_risk", self.risk_manager.max_open_risk),
            )
        )
        self.risk_manager.max_trades_per_day = int(
            risk_cfg.get(
                "max_trades_per_day",
                config.get("max_trades_per_day", self.risk_manager.max_trades_per_day),
            )
        )
        self._set_risk_mode(str(risk_cfg.get("risk_mode", config.get("risk_mode", self.risk_manager.risk_mode))))
        self.risk_manager.max_notional_per_symbol = float(
            risk_cfg.get(
                "max_notional_per_symbol",
                config.get("max_notional_per_symbol", self.risk_manager.max_notional_per_symbol),
            )
        )

        # PRD 9.2: Propagate parameter updates to existing strategies
        strategies_cfg = config.get("strategies", {})
        if not isinstance(strategies_cfg, dict):
            strategies_cfg = {}

        for name, strat in self.strategies.items():
            s_cfg = strategies_cfg.get(name)
            if isinstance(s_cfg, dict):
                try:
                    strat.update_params(s_cfg)
                except Exception as e:
                    self.logger.error("Failed to update strategy params", strategy=name, error=str(e))

        self._refresh_strategy_engine()

    def _refresh_strategy_engine(self) -> None:
        """Rebuild strategy engine with current config and routing."""
        from src.core.domain import Regime

        cfg = self.config or {}
        strategies_cfg: Dict[str, Any] = (
            cfg.get("strategies") if isinstance(cfg.get("strategies"), dict) else {}
        ) or {}

        activation_policies = build_activation_policies_from_config(strategies_cfg)

        # Build legacy strategies_by_regime from strategy_routing config, filtering out
        # strategies that have regimes.{regime}.enabled: False in their strategy config.
        routing_cfg = cfg.get("strategy_routing", {})
        strategies_by_regime: Dict[Regime, List[str]] = {}
        if isinstance(routing_cfg, dict):
            for regime_str, strat_list in routing_cfg.items():
                try:
                    regime = Regime(regime_str.lower())
                except ValueError:
                    continue
                if not isinstance(strat_list, list):
                    continue
                enabled_strats: List[str] = []
                for strat_name in strat_list:
                    strat_cfg = strategies_cfg.get(strat_name, {})
                    regimes_cfg = strat_cfg.get("regimes", {}) if isinstance(strat_cfg, dict) else {}
                    regime_override = regimes_cfg.get(regime_str.lower(), {}) if isinstance(regimes_cfg, dict) else {}
                    if isinstance(regime_override, dict) and not regime_override.get("enabled", True):
                        continue
                    enabled_strats.append(strat_name)
                strategies_by_regime[regime] = enabled_strats

        routing = StrategyRouting(
            strategies_by_regime=strategies_by_regime,
            activation_policies=activation_policies,
        )
        self.strategy_engine = StrategyEngine(
            self.strategies,
            routing,
            self.logger,
            on_error=lambda: self._inc_error("strategy"),
        )

    def on_bar(self, symbol_or_bar: Any, bar: Optional[Any] = None) -> None:
        """
        Process bar data for symbol (main trading loop entry point).

        Primary bar processing pipeline processing OHLCV data, updating technical indicators,
        evaluating strategy signals, checking exits, and managing scanner-based watchlist updates.
        This is the core heartbeat of the trading engine, called for each new bar.

        Workflow:
        1. Normalize bar data (handle both symbol+bar or single bar object)
        2. Update market state with current bar timestamp
        3. Update symbol state and technical indicator cache
        4. Run strategies for signal generation
        5. Manage positions (exit checks via PositionManager.on_bar())
        6. Process scanner results if interval reached
        7. Increment bars_processed counter

        Args:
            symbol_or_bar: Either symbol string OR bar object (backward compatibility)
            bar: Bar data or None. If symbol_or_bar is bar object, this is ignored.
                 Bar contains: timestamp/time, open, high, low, close, volume, symbol

        Returns:
            None

        Side Effects:
            - Updates market_state.time and regime
            - Updates symbol_state and indicator cache
            - Calls PositionManager.on_bar() for exit checks
            - Routes to StrategyEngine.on_bar() for signal generation
            - Processes signals (riskchecks, order submission)
            - Triggers scanner refresh if interval reached
            - Increments bars_processed counter
            - Logs bar processing, latency metrics, and errors

        Note:
            - PRD 6.2: WebSocket handlers call on_bar(bar) or on_bar(symbol, bar)
            - PRD 6.4: Strategy routing by market regime
            - Scanner refresh happens every N bars (scanner_interval config)
            - Best-effort processing (errors logged, not raised)
            - Latency metrics logged for bar processing performance
        """
        if bar is None:
            bar = symbol_or_bar
            symbol = getattr(bar, "symbol", None) or getattr(bar, "S", None)
            if not symbol:
                raise ValueError("on_bar(bar) requires bar.symbol")
            symbol = str(symbol)
        else:
            symbol = str(symbol_or_bar)

        start_bar = _time.perf_counter()

        self.bars_processed += 1
        bar_time = getattr(bar, "time", getattr(bar, "timestamp", None))

        try:
            # In backtest mode, skip logger bind overhead (creates new object per bar)
            if self.backtest_mode:
                log = self.logger
            else:
                _bind = getattr(self.logger, "bind", None)
                log = _bind(symbol=symbol, run_id=self.run_id) if callable(_bind) else self.logger

            # Helper: Update State
            state = self._update_symbol_state(symbol, bar, bar_time)

            # Helper: Manage Positions (exits first to avoid conflicting entry + exit on same bar)
            self._manage_positions(symbol, state, log)

            # Helper: Run Strategies (after exits, so new entries don't conflict with pending exits)
            self._run_strategies(symbol, bar, state, log)

            # Emit periodic health metrics based on bar time (deterministic for replay).
            # Skip in backtest mode to avoid per-bar datetime arithmetic.
            if not self.backtest_mode:
                try:
                    self._maybe_log_health(self.market_state.time)
                except Exception:
                    self._inc_error("execution")

            # PRD 11.2: latency observability (skip in backtest mode).
            if not self.backtest_mode:
                try:
                    latency_ms = (_time.perf_counter() - start_bar) * 1000.0
                    max_ms = float(self.config.get("max_bar_latency_ms", 1000.0) or 1000.0)
                    if latency_ms > max_ms:
                        log.warning(
                            "Slow bar processing",
                            bar_latency_ms=float(latency_ms),
                            max_bar_latency_ms=float(max_ms),
                            bars_processed=self.bars_processed,
                            run_id=self.run_id,
                        )
                except Exception as e:
                    self.logger.debug(
                        "Latency logging failed",
                        operation="log_bar_latency",
                        error=str(e),
                    )
        except Exception as e:
            self.consecutive_on_bar_errors += 1
            self._inc_error("execution")

            _bind = getattr(self.logger, "bind", None)
            log = _bind(symbol=symbol, run_id=self.run_id) if callable(_bind) else self.logger
            log.error(
                "Bar processing failed",
                error_code=ErrorCode.ENGINE_ON_BAR_FAILED.value,
                symbol=symbol,
                bar_time=str(bar_time or self.clock()),
                regime=getattr(self.market_state.regime, "value", str(self.market_state.regime)),
                error=str(e),
                consecutive_errors=self.consecutive_on_bar_errors,
                run_id=self.run_id,
                exc_info=True,
            )

            if self.consecutive_on_bar_errors >= self.max_consecutive_errors:
                log.error(
                    "CRITICAL: Max consecutive execution errors exceeded. Crashing process.",
                    max_consecutive_errors=self.max_consecutive_errors,
                )
                raise RuntimeError("Max consecutive execution errors exceeded") from e
            return

        # If we reached here, the bar was processed successfully (or minor errors were suppressed)
        self.consecutive_on_bar_errors = 0

    def _update_symbol_state(self, symbol: str, bar: Any, bar_time: Any) -> SymbolState:
        """Update symbol state with new bar data."""
        # Handle index symbol updates (regime, cache clearing)
        if symbol == self.config.get("index_symbol", "SPY"):
            self._handle_index_bar_update(bar, bar_time)

        # Handle vol symbol updates (VXX for risk axis)
        regime_cfg = self.config.get("regime", {}) or {}
        vol_symbol = regime_cfg.get("vol_symbol")
        if vol_symbol and symbol == str(vol_symbol).upper():
            try:
                self.market_manager.update_vol(bar)
            except Exception as e:
                self.logger.error(
                    "Vol symbol update failed — regime volatility axis may be stale",
                    symbol=symbol,
                    error=str(e),
                    exc_info=True,
                )

        # Initialize or get symbol state
        state = self._get_or_create_symbol_state(symbol)
        state.bars.append(bar)

        # Update VWAP tracking
        self._update_session_vwap(state, bar, bar_time)

        # Update indicator cache
        try:
            self._update_indicator_cache(state, bar)
        except Exception as e:
            self._inc_error("execution")
            self.logger.error(
                "Indicator cache update failed — strategies may use stale values",
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )

        # Update unrealized PnL
        try:
            mark = float(getattr(bar, "close", 0.0) or 0.0)
            self.position_manager.update_unrealized_pnl(state, mark)
        except Exception as e:
            self._inc_error("execution")
            self.logger.error(
                "Unrealized PnL update failed — position sizing may use stale P&L",
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )

        return state

    def _get_or_create_symbol_state(self, symbol: str) -> SymbolState:
        """Get existing or create new symbol state."""
        if symbol not in self.symbol_states:
            # In backtest mode, set scanner_bypass so V2 strategies with
            # activation policies can fire without a live HMM regime detector.
            meta = {}
            if getattr(self, "backtest_mode", False):
                meta["scanner_bypass"] = True
            self.symbol_states[symbol] = SymbolState(
                symbol=symbol,
                bars=deque(maxlen=500),
                position=None,
                indicators={},
                open_orders={},
                allowed_strategies=list(self.strategies.keys()),
                meta=meta,
            )
        return self.symbol_states[symbol]

    def _extract_current_date(self, bar_time: Any) -> Optional[str]:
        """Extract current date with fallback to clock."""
        try:
            time_source = bar_time or self.clock()
            if hasattr(time_source, "date"):
                return time_source.date().isoformat()
            self.logger.debug(
                "bar_time lacks date() method, using clock fallback",
                bar_time_type=type(bar_time).__name__,
            )
            return self.clock().date().isoformat()
        except Exception:
            self.logger.warning(
                "Failed to extract date from bar_time, using clock fallback",
                bar_time=str(bar_time),
                exc_info=True,
            )
            try:
                return self.clock().date().isoformat()
            except Exception:
                return None

    def _handle_index_bar_update(self, bar: Any, bar_time: Any) -> None:
        """Handle index symbol updates: regime tracking and cache clearing."""
        current_date = self._extract_current_date(bar_time)

        # Update market state first so regime comparisons reflect this bar.
        index_symbol = self.config.get("index_symbol", "SPY")
        self.market_manager.update(bar, index_symbol=index_symbol)

        current_regime = self.market_manager.state.regime

        # Clear on regime change
        if self._last_regime and current_regime != self._last_regime:
            # Get current regime snapshot for 5-axis info
            snapshot = self.market_manager.state.regime_snapshot
            regime_tags = snapshot.regime_tags if snapshot else {}
            self.logger.info(
                "Market regime changed, clearing feature cache",
                old_regime=(self._last_regime.value if hasattr(self._last_regime, "value") else str(self._last_regime)),
                new_regime=(current_regime.value if hasattr(current_regime, "value") else str(current_regime)),
                regime_tags=regime_tags,
            )
            self._clear_all_features()

        # Clear daily (safety net)
        if current_date and self._last_cache_clear_date != current_date:
            self.logger.debug("Daily feature cache clear", date=current_date)
            self._clear_all_features()
            self._last_cache_clear_date = current_date

        # Update regime tracking
        self._last_regime = current_regime

    def _update_session_vwap(self, state: SymbolState, bar: Any, bar_time: Any) -> None:
        """Update session VWAP tracking and daily high/low for intraday calculations."""
        try:
            bt = bar_time or self.clock()
            session_day = bt.date() if hasattr(bt, "date") else None
            prev_day = state.indicators.get("session_day")

            # Reset on new session
            if session_day is not None and prev_day != session_day:
                # Save completed day's high/low before resetting
                if prev_day is not None:
                    daily_highs = state.indicators.get("daily_highs", deque(maxlen=20))
                    daily_lows = state.indicators.get("daily_lows", deque(maxlen=20))
                    cur_h = state.indicators.get("current_day_high")
                    cur_l = state.indicators.get("current_day_low")

                    if cur_h is not None and cur_l is not None:
                        daily_highs.append(cur_h)
                        daily_lows.append(cur_l)
                        state.indicators["daily_highs"] = daily_highs
                        state.indicators["daily_lows"] = daily_lows

                        # Set convenience properties
                        state.meta["prior_day_high"] = cur_h
                        state.meta["prior_day_low"] = cur_l

                # Initialize new session
                state.indicators["session_day"] = session_day
                state.indicators["cum_pv"] = 0.0
                state.indicators["cum_vol"] = 0.0
                state.indicators["current_day_high"] = float(getattr(bar, "high", 0.0) or 0.0)
                state.indicators["current_day_low"] = float(getattr(bar, "low", 0.0) or 0.0)
            else:
                # Update current day high/low
                h = float(getattr(bar, "high", 0.0) or 0.0)
                low_val = float(getattr(bar, "low", 0.0) or 0.0)
                cur_h = state.indicators.get("current_day_high")
                cur_l = state.indicators.get("current_day_low")

                if cur_h is None or h > cur_h:
                    state.indicators["current_day_high"] = h
                if cur_l is None or low_val < cur_l:
                    state.indicators["current_day_low"] = low_val

            # Calculate cumulative values
            vol = float(getattr(bar, "volume", 0.0) or 0.0)
            tp = (
                float(getattr(bar, "high", 0.0) or 0.0)
                + float(getattr(bar, "low", 0.0) or 0.0)
                + float(getattr(bar, "close", 0.0) or 0.0)
            ) / 3.0
            cum_pv = float(state.indicators.get("cum_pv", 0.0) or 0.0) + (tp * vol)
            cum_vol = float(state.indicators.get("cum_vol", 0.0) or 0.0) + vol
            state.indicators["cum_pv"] = cum_pv
            state.indicators["cum_vol"] = cum_vol

            # Calculate and store VWAP
            if cum_vol > 0:
                session_vwap = cum_pv / cum_vol
                state.indicators["session_vwap"] = session_vwap
                try:
                    bar.vwap = session_vwap
                except Exception as e:
                    self.logger.error(
                        "Session VWAP bar assignment failed",
                        symbol=getattr(state, "symbol", "unknown"),
                        error=str(e),
                        exc_info=True,
                    )
        except Exception as e:
            self._inc_error("execution")
            self.logger.error(
                "Session VWAP calculation failed — VWAP-dependent strategies may use stale values",
                symbol=getattr(state, "symbol", "unknown"),
                error=str(e),
                exc_info=True,
            )

    def _run_strategies(self, symbol: str, bar: Any, state: SymbolState, log: Any) -> None:
        # Run Strategies via StrategyEngine (PRD 6.4)
        if self.strategy_engine is None:
            self._refresh_strategy_engine()

        if self.strategy_engine is not None:
            signals = self.strategy_engine.on_bar(symbol, bar, state, self.market_state)

            for sig in signals:
                s_bind = getattr(log, "bind", None)
                slog = (
                    s_bind(
                        strategy=sig.strategy,
                        correlation_id=getattr(sig, "correlation_id", "") or None,
                        regime_tags=sig.regime_tags,
                    )
                    if callable(s_bind)
                    else log
                )
                slog.info(
                    "Signal generated",
                    symbol=sig.symbol,
                    strategy=sig.strategy,
                    regime_tags=sig.regime_tags,
                    correlation_id=getattr(sig, "correlation_id", "") or None,
                    signal=sig,
                    run_id=self.run_id,
                )
                self.signals_generated += 1

                # PRD Phase 6: Meta-Labeling Vetting
                if self.meta_labeler.vet_signal(sig):
                    self._process_signal(sig)
                else:
                    slog.info(
                        "Signal rejected by meta-labeler v1",
                        symbol=sig.symbol,
                        strategy=sig.strategy,
                    )

    def _manage_positions(self, symbol: str, state: SymbolState, log: Any) -> None:
        # PositionManager exit checks (PRD 6.7 best-effort)
        try:
            decision = self.position_manager.on_bar(
                state,
                self.market_state,
                broker_managed_exits=bool(getattr(self.order_executor, "broker_managed_exits", False)),
            )
            if decision.intent is not None:
                e_bind = getattr(log, "bind", None)
                elog = (
                    e_bind(
                        strategy=decision.intent.strategy,
                        correlation_id=decision.intent.correlation_id,
                    )
                    if callable(e_bind)
                    else log
                )
                elog.info(
                    "Exit generated",
                    symbol=state.symbol,
                    strategy=decision.intent.strategy,
                    correlation_id=decision.intent.correlation_id,
                    reason=decision.reason,
                )
                if self.order_executor:
                    self.order_executor.submit(decision.intent)
                    self.orders_submitted += 1
        except Exception as e:
            self._inc_error("orders")

            self.logger.error(
                "PositionManager exit failed",
                error_code=ErrorCode.POSITION_EXIT_FAILED.value,
                symbol=symbol,
                regime=getattr(self.market_state.regime, "value", str(self.market_state.regime)),
                error=str(e),
                exc_info=True,
            )

    def _process_signal(self, signal: Signal):
        """
        Process strategy signal through risk checks and route to order execution.

        Core signal processing pipeline bridging strategy logic and order placement.
        """
        log = self._bind_signal_logger(signal)
        log.info(
            "Processing signal",
            symbol=signal.symbol,
            strategy=signal.strategy,
            correlation_id=signal.correlation_id,
            regime=signal.regime_tags,
            run_id=self.run_id,
        )

        # 1. Risk Check
        active_positions = [s.position for s in self.symbol_states.values() if s.position is not None]

        start_risk = _time.perf_counter()
        try:
            intents = self.risk_manager.apply(
                signal,
                self.symbol_states[signal.symbol],
                self.market_state,
                current_positions=active_positions,
                account_equity=min(float(getattr(self.account, "equity", 0.0) or 0.0), 100_000.0),
            )
        except Exception as e:
            self._log_risk_failure(log, signal, e)
            return
        risk_latency = (_time.perf_counter() - start_risk) * 1000

        # 2. Persist Signal
        self._persist_signal(signal, intents)

        if not intents:
            log.info(
                "Signal rejected by Risk Manager",
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=signal.correlation_id,
                rejection_reason=self.risk_manager.last_rejection_reason or None,
                run_id=self.run_id,
            )
            return

        # 3. Store pending entry context
        self._store_pending_entry(signal, intents)

        # 4. Order Execution
        self._execute_signal_intents(signal, intents, log, risk_latency)

    def _bind_signal_logger(self, signal: Signal) -> Any:
        """Bind logger with signal context."""
        _bind = getattr(self.logger, "bind", None)
        return (
            _bind(
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=signal.correlation_id,
                run_id=self.run_id,
                regime=signal.regime_tags,
            )
            if callable(_bind)
            else self.logger
        )

    def _log_risk_failure(self, log: Any, signal: Signal, error: Exception) -> None:
        """Log risk manager failure."""
        self._inc_error("risk")

        log.error(
            "RiskManager.apply failed; dropping signal",
            error_code=ErrorCode.RISK_APPLY_FAILED.value,
            symbol=signal.symbol,
            strategy=signal.strategy,
            correlation_id=getattr(signal, "correlation_id", "") or None,
            error=str(error),
            run_id=self.run_id,
        )

    def _persist_signal(self, signal: Signal, intents: List[Any]) -> None:
        """Persist signal to database."""
        if not self.db or self.backtest_mode:
            return

        rejection_reason = None
        if not intents:
            rejection_reason = self.risk_manager.last_rejection_reason or "RISK_REJECTED"

        ok = self.db.write(
            "signal",
            lambda session: session.add(
                DbSignal(
                    correlation_id=signal.correlation_id,
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    regime=str(signal.regime_tags),
                    time=signal.generated_at,
                    raw_side=signal.side.value,
                    raw_size=signal.size_hint,
                    accepted=bool(intents),
                    rejection_reason=rejection_reason,
                    meta_json=self._sanitize_features_snapshot(signal.meta),
                    feature_snapshot_json=(
                        self._sanitize_features_snapshot(asdict(signal.feature_snapshot))
                        if signal.feature_snapshot
                        else None
                    ),
                )
            ),
        )
        if not ok:
            self._inc_error("db")

    def _store_pending_entry(self, signal: Signal, intents: List[Any]) -> None:
        """Store pending entry context for fill correlation."""
        if not intents:
            return

        intent0 = intents[0]
        state = self.symbol_states[signal.symbol]
        pending = state.meta.setdefault("pending_entries", {})

        if not isinstance(pending, dict) or not hasattr(intent0, "qty"):
            return

        # Strategy's per-signal exit config (primary source)
        sig_exit_cfg = (signal.meta or {}).get("exit_config", {}) if signal.meta else {}
        if not isinstance(sig_exit_cfg, dict):
            sig_exit_cfg = {}

        # max_hold: prefer signal exit_config → fallback to YAML strategy config
        max_hold_seconds = None
        sig_max_hold = sig_exit_cfg.get("max_hold_minutes")
        if sig_max_hold is not None:
            max_hold_seconds = int(float(sig_max_hold) * 60)
        else:
            max_hold_seconds = self._get_max_hold_seconds(signal.strategy)

        # Get global advanced exits config (fallback for trailing)
        adv_exits = self._get_advanced_exits_config()

        # trailing: prefer signal exit_config → fallback to global risk config
        trailing_enabled = sig_exit_cfg.get(
            "trailing_enabled",
            adv_exits.get("trailing_stop", {}).get("enabled", False),
        )
        trailing_pct = sig_exit_cfg.get(
            "trail_pct",
            adv_exits.get("trailing_stop", {}).get("trail_pct", 0.02),
        )

        # Calculate regime-aware stop multiplier
        regime_stop_mult = self._get_regime_stop_multiplier(signal.regime_tags, adv_exits)

        # Prepare features payload
        features_payload = state.meta.get("features_snapshot")
        if isinstance(features_payload, dict):
            features_payload = self._sanitize_features_snapshot(features_payload)

        pending[signal.correlation_id] = {
            "strategy": signal.strategy,
            "regime_tags": signal.regime_tags,
            "entry_price": signal.entry_price,
            "stop_price": signal.stop_price,
            "target_price": signal.target_price,
            "qty": intent0.qty,
            "open_risk": abs(signal.entry_price - signal.stop_price) * float(intent0.qty),
            "features": (
                {
                    **(features_payload or {}),
                    "scanner_score": state.meta.get("score"),
                }
                if isinstance(features_payload, dict)
                else features_payload
            ),
            "entry_time": signal.generated_at,
            "max_hold_seconds": max_hold_seconds,
            # Advanced exit fields
            "trailing_stop_enabled": trailing_enabled,
            "trailing_stop_pct": trailing_pct if trailing_enabled else None,
            "regime_stop_multiplier": regime_stop_mult,
            "partial_exit_levels": sig_exit_cfg.get("partial_exits", []),
            "trail_min_profit_r": sig_exit_cfg.get("trail_min_profit_r"),
        }

    def _get_max_hold_seconds(self, strategy_name: str) -> Optional[int]:
        """Get max hold seconds from strategy config."""
        strat_cfg = None
        if isinstance(self.config.get("strategies"), dict):
            strat_cfg = (self.config.get("strategies") or {}).get(strategy_name)
        max_hold_minutes = None
        if isinstance(strat_cfg, dict):
            max_hold_minutes = strat_cfg.get("max_hold_minutes")
        if max_hold_minutes is not None:
            return int(float(max_hold_minutes) * 60)
        return None

    def _get_advanced_exits_config(self) -> Dict[str, Any]:
        """Get advanced exits configuration from risk config."""
        risk_cfg = self.config.get("risk", {})
        if not isinstance(risk_cfg, dict):
            risk_cfg = {}

        adv_exits = risk_cfg.get("advanced_exits", {})
        if not isinstance(adv_exits, dict):
            adv_exits = {}

        return adv_exits

    def _get_regime_stop_multiplier(self, regime_tags: Optional[Dict[str, str]], adv_exits: Dict[str, Any]) -> float:
        """Calculate stop width multiplier based on volatility regime."""
        if not regime_tags or not adv_exits.get("regime_aware_stops", False):
            return 1.0

        vol_regime = regime_tags.get("vol", "normal")
        multipliers = adv_exits.get(
            "regime_stop_multipliers",
            {"low": 0.75, "normal": 1.0, "high": 1.5, "shock": 2.0},
        )
        if not isinstance(multipliers, dict):
            return 1.0

        return float(multipliers.get(vol_regime, 1.0))

    def _execute_signal_intents(self, signal: Signal, intents: List[Any], log: Any, risk_latency: float) -> None:
        """Execute order intents for a signal."""
        if not self.order_executor:
            self.logger.warning(
                "No OrderExecutor available (paper/mock mode)",
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=signal.correlation_id,
                intents=intents,
                run_id=self.run_id,
            )
            return

        # Check DB trading halt conditions
        if self._should_halt_trading_for_db(log):
            return

        # Submit each intent
        for intent in intents:
            self._submit_single_intent(intent, log, risk_latency, _time.perf_counter)

    def _should_halt_trading_for_db(self, log: Any) -> bool:
        """Check if trading should halt due to DB failures."""
        if not self.db:
            return False

        mode = str((self.config.get("db_trading_mode") or "warn")).lower()
        if mode not in ("halt", "stop"):
            return False

        buf_len = 0
        buf_max = 0
        try:
            buf_len = int(self.db.write_buffer_len())
            buf_max = int(self.db.write_buffer_max())
        except Exception:
            log.warning("db_buffer_metrics_unavailable", exc_info=True)

        threshold = int(self.config.get("db_trading_halt_buffer_len", max(1, buf_max // 2 or 1)))

        if self.db.last_db_write_error and buf_len >= threshold:
            self._set_risk_mode("off")
            log.error(
                "Halting trading due to persistent DB failures",
                last_db_error=self.db.last_db_write_error,
                db_buffer_len=buf_len,
                db_buffer_max=buf_max,
                threshold=threshold,
            )
            return True

        return False

    def _submit_single_intent(self, intent: Any, log: Any, risk_latency: float, perf_counter: Any) -> None:
        """Submit a single order intent."""
        assert self.order_executor is not None
        try:
            start_exec = perf_counter()
            self.order_executor.submit(intent)
            exec_latency = (perf_counter() - start_exec) * 1000
            self.orders_submitted += 1
            log.info(
                "Order submitted",
                symbol=intent.symbol,
                strategy=intent.strategy,
                correlation_id=intent.correlation_id,
                intent=intent,
                risk_latency_ms=risk_latency,
                exec_latency_ms=exec_latency,
                run_id=self.run_id,
            )
        except Exception as e:
            self._inc_error("orders")
            self.logger.error(
                "Order execution failed",
                symbol=intent.symbol,
                strategy=intent.strategy,
                correlation_id=intent.correlation_id,
                error=str(e),
                run_id=self.run_id,
                exc_info=True,
            )

    def on_fill(self, fill: Dict[str, Any]):
        """Process fill event and update position state."""
        symbol = fill["symbol"]
        if symbol not in self.symbol_states:
            self.logger.warning("Fill received for unknown symbol", symbol=symbol)
            return

        state = self.symbol_states[symbol]
        fill_ts = fill.get("timestamp") or self.market_state.time

        # Normalize correlation ID
        fill = self._normalize_fill_correlation_id(fill, symbol, fill_ts)
        corr_id = fill["correlation_id"]

        # Dedup guard: skip if this exact fill was already processed (e.g. on reconnect/restart).
        # Use a composite key so entry and exit fills sharing the same correlation_id
        # are treated as distinct events. Pre-populated from DB on startup.
        fill_key = f"{corr_id}|{fill.get('side', '')}|{fill.get('price', '')}|{fill.get('qty', '')}"
        if fill_key in self._processed_fill_ids:
            self.logger.debug(
                "Duplicate fill skipped",
                symbol=symbol,
                correlation_id=corr_id,
                fill_key=fill_key,
            )
            return
        self._processed_fill_ids.add(fill_key)
        if len(self._processed_fill_ids) > 10_000:
            # Evict oldest half instead of clearing all — prevents reprocessing window
            evict_list = list(self._processed_fill_ids)[:5_000]
            self._processed_fill_ids -= set(evict_list)

        # Persist fill
        if self.db:
            self._persist_fill(
                fill,
                corr_id,
                fill_ts,
                float(fill["price"]),
                float(fill["qty"]),
            )

        # Process via PositionManager
        decision = self._process_fill_with_position_manager(state, fill)
        if decision is None:
            return

        self._log_position_decision(decision, state, fill)
        self._update_realized_pnl(decision, fill_ts)

        # Handle closed trade
        if decision.closed_trade is not None:
            self._handle_closed_trade(decision.closed_trade)

    def _normalize_fill_correlation_id(self, fill: Dict[str, Any], symbol: str, fill_ts: Any) -> Dict[str, Any]:
        """Normalize correlation ID with fallbacks."""
        corr = str(fill.get("correlation_id") or "")
        corr_id = corr

        if not corr_id:
            broker_order_id = fill.get("broker_order_id")
            if broker_order_id is not None and str(broker_order_id):
                corr_id = f"alpaca-{str(broker_order_id)}"
            else:
                import hashlib

                seed = f"{symbol}|{fill.get('side', '')}|{fill_ts}"
                corr_id = "fill-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

        fill = dict(fill)
        fill["correlation_id"] = corr_id
        return fill

    def _process_fill_with_position_manager(self, state: SymbolState, fill: Dict[str, Any]) -> Optional[Any]:
        """Process fill through PositionManager."""
        try:
            return self.position_manager.on_fill(
                state,
                self.market_state,
                fill,
                risk_cfg=(self.config.get("risk") if isinstance(self.config.get("risk"), dict) else None),
            )
        except Exception as e:
            self._inc_error("execution")

            self.logger.error(
                "PositionManager on_fill failed",
                error_code=ErrorCode.POSITION_UPDATE_FAILED.value,
                symbol=fill["symbol"],
                correlation_id=fill.get("correlation_id"),
                error=str(e),
                exc_info=True,
            )
            return None

    def _handle_closed_trade(self, closed: Any) -> None:
        """Handle a closed trade: persist, record, and track."""
        try:
            self.closed_trades.append(closed)
        except Exception as e:
            self.logger.error(
                "Failed to append closed trade",
                error=str(e),
                exc_info=True,
            )

        if self.db:
            self._persist_closed_trade(closed)

        try:
            self.risk_manager.record_completed_trade(closed.strategy, pnl_net=closed.pnl_net, as_of=closed.exit_time)
        except Exception as e:
            self.logger.error(
                "Risk manager failed to record completed trade",
                strategy=getattr(closed, "strategy", "unknown"),
                error=str(e),
                exc_info=True,
            )

        # Reset pair_trading_v2 signal_active flag so the pair can re-enter
        self._maybe_reset_pair_signal(closed)

    def _maybe_reset_pair_signal(self, closed: Any) -> None:
        """Reset pair_trading_v2 signal_active when a pair leg position closes."""
        try:
            strategy_name = getattr(closed, "strategy", None)
            if strategy_name != "pair_trading_v2":
                return
            strat = self.strategies.get("pair_trading_v2")
            if strat is None:
                return
            symbol = getattr(closed, "symbol", None)
            if not symbol:
                return
            pair_keys = getattr(strat, "_symbol_to_pairs", {}).get(symbol, [])
            for key in pair_keys:
                strat.mark_pair_inactive(key)
                self.logger.debug(
                    "Pair signal_active reset after position close",
                    pair=key,
                    symbol=symbol,
                )
        except Exception as e:
            self.logger.error(
                "Pair signal reset failed — pair may not re-enter after position close",
                symbol=getattr(closed, "symbol", "unknown"),
                error=str(e),
                exc_info=True,
            )

    def _restore_fill_dedup_from_db(self) -> None:
        """Pre-populate fill dedup set from recent DB fills to survive restarts.

        Without this, a restart during market hours could reprocess fills received
        during recovery, creating phantom positions (Finding 4, consensus 4/8).
        """
        if not self.db:
            return
        try:
            from src.analysis.schema import Fill as DbFill
            from src.analysis.schema import Order as DbOrder

            with self.db.get_session() as session:
                # Load last 500 fills (covers a full trading day) to rebuild dedup keys
                recent_fills = (
                    session.query(DbFill, DbOrder)
                    .join(DbOrder, DbFill.order_id == DbOrder.id)
                    .order_by(DbFill.id.desc())
                    .limit(500)
                    .all()
                )
                for fill_row, order_row in recent_fills:
                    corr_id = getattr(order_row, "correlation_id", "") or ""
                    side = getattr(order_row, "side", "") or ""
                    price = round(getattr(fill_row, "fill_price", 0.0), 4)
                    qty = round(getattr(fill_row, "fill_qty", 0.0), 4)
                    fill_key = f"{corr_id}|{side}|{price}|{qty}"
                    self._processed_fill_ids.add(fill_key)

                if self._processed_fill_ids:
                    self.logger.info(
                        "Restored fill dedup set from DB",
                        count=len(self._processed_fill_ids),
                    )
        except Exception as e:
            self.logger.warning(
                "Could not restore fill dedup from DB — dedup starts empty",
                error=str(e),
            )

    def _persist_fill(
        self,
        fill: Dict[str, Any],
        corr_id: str,
        fill_ts: datetime,
        fill_price: float,
        fill_qty: float,
    ) -> None:
        fill_ts = self._normalize_timestamp(fill_ts)
        broker_order_id = fill.get("broker_order_id")
        explicit_order_id = fill.get("order_id")
        symbol = fill["symbol"]

        def _write_fill(session: Session) -> None:
            order_id: Optional[int] = None
            if explicit_order_id is not None:
                order_id = int(explicit_order_id)
            elif broker_order_id is not None:
                order_row = (
                    session.query(DbOrder)
                    .filter(DbOrder.broker_order_id == str(broker_order_id))
                    .order_by(DbOrder.id.desc())
                    .first()
                )
                if order_row is None:
                    # Create a minimal order row so fills are never dropped.
                    session.add(
                        DbOrder(
                            correlation_id=corr_id,
                            symbol=symbol,
                            side=str(fill.get("side", "") or ""),
                            qty=float(fill_qty),
                            type="unknown",
                            limit_price=None,
                            status="filled",
                            time_placed=self._normalize_timestamp(fill_ts),
                            time_last_update=self._normalize_timestamp(fill_ts),
                            broker_order_id=str(broker_order_id),
                            meta_json={"recovered_from_fill": True},
                        )
                    )
                    session.flush()
                    order_row = (
                        session.query(DbOrder)
                        .filter(DbOrder.broker_order_id == str(broker_order_id))
                        .order_by(DbOrder.id.desc())
                        .first()
                    )
                if order_row is not None and order_row.id is not None:
                    order_id = int(order_row.id)

            if order_id is None:
                return

            # Idempotency — round floats to avoid IEEE 754 mismatch on dedup
            rounded_price = round(fill_price, 4)
            rounded_qty = round(fill_qty, 4)
            existing = (
                session.query(DbFill)
                .filter(DbFill.order_id == int(order_id))
                .filter(DbFill.fill_time == fill_ts)
                .filter(DbFill.fill_price.between(rounded_price - 1e-6, rounded_price + 1e-6))
                .filter(DbFill.fill_qty.between(rounded_qty - 1e-6, rounded_qty + 1e-6))
                .first()
            )
            if existing is not None:
                return

            session.add(
                DbFill(
                    order_id=order_id,
                    fill_time=fill_ts,
                    fill_price=rounded_price,
                    fill_qty=rounded_qty,
                )
            )

        if self.db:
            self.db.write("fill", _write_fill)

    def _log_position_decision(self, decision: Any, state: SymbolState, fill: Dict[str, Any]) -> None:
        if decision.event == "opened":
            self.logger.info("Position opened", position=state.position)
        elif decision.event == "increased":
            self.logger.info("Position increased", position=state.position)
        elif decision.event == "reduced":
            self.logger.info(
                "Position reduced",
                position=state.position,
                pnl=decision.realized_pnl_delta,
            )
        elif decision.event == "closed":
            self.logger.info(
                "Position closed",
                pnl=decision.realized_pnl_delta,
                strategy=(
                    decision.closed_trade.strategy if decision.closed_trade is not None else fill.get("strategy")
                ),
            )

    def _update_realized_pnl(self, decision: Any, fill_ts: Any) -> None:
        if decision.realized_pnl_delta:
            try:
                self.risk_manager.update_pnl(
                    float(decision.realized_pnl_delta),
                    as_of=(fill_ts if isinstance(fill_ts, datetime) else None),
                )
            except Exception as e:
                self.logger.error(
                    "Risk manager PnL update failed — daily loss limit may not trigger",
                    pnl_delta=float(decision.realized_pnl_delta),
                    error=str(e),
                    exc_info=True,
                )
            try:
                self.market_state.daily_pnl = float(self.risk_manager.current_daily_pnl)
            except Exception as e:
                self.logger.error(
                    "Market state daily PnL update failed — risk dashboard may show stale P&L",
                    error=str(e),
                    exc_info=True,
                )

    @staticmethod
    def _normalize_timestamp(ts: Any) -> Any:
        """Convert Pandas Timestamps / nanosecond ints to Python datetime for SQLite."""
        if ts is None:
            return ts
        if isinstance(ts, (int, float)):
            from datetime import datetime as _dt
            from datetime import timezone as _tz

            return _dt.fromtimestamp(ts / 1e9, tz=_tz.utc)
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
        return ts

    def _persist_closed_trade(self, closed: Any) -> None:
        def _write_trade(session: Session) -> None:
            trade = DbTrade(
                symbol=closed.symbol,
                strategy=closed.strategy,
                regime_at_entry=str(closed.regime_tags_at_entry),
                regime_at_exit=str(closed.regime_tags_at_exit),
                side=closed.side,
                qty=closed.qty,
                entry_time=self._normalize_timestamp(closed.entry_time),
                exit_time=self._normalize_timestamp(closed.exit_time),
                entry_price=closed.entry_price,
                exit_price=closed.exit_price,
                pnl_gross=closed.pnl_gross,
                pnl_net=closed.pnl_net,
                initial_risk=closed.initial_risk,
                mae_r=closed.mae_r,
                mfe_r=closed.mfe_r,
                commission=closed.commission,
                slippage_estimate=closed.slippage_estimate,
                pnl_r=closed.pnl_r,
                holding_period_seconds=closed.holding_period_seconds,
                features_json=closed.features_json,
                correlation_id=closed.correlation_id,
            )
            session.add(trade)
            session.flush()

            if trade.id is not None and closed.correlation_id:
                (
                    session.query(DbOrder)
                    .filter(DbOrder.correlation_id == closed.correlation_id)
                    .update({"trade_id": int(trade.id)})
                )

        if self.db:
            self.db.write("trade", _write_trade)

    async def on_trade_update(self, update: Any) -> None:
        """
        Handles Alpaca trade updates (order lifecycle + fills).

        This is called by `AlpacaClient.start_trade_stream`.
        """
        if not self.order_executor:
            self.logger.warning("Trade update received but no OrderExecutor configured")
            return

        try:
            info = self.order_executor.handle_trade_update(update)
        except Exception as e:
            self.error_counts["orders"] += 1

            self.logger.error(
                "Trade update handling failed",
                error_code=ErrorCode.TRADE_UPDATE_FAILED.value,
                error=str(e),
            )
            return

        event = str(info.get("event", "") or "")
        symbol = str(info.get("symbol", "") or "")
        side = str(info.get("side", "") or "")
        broker_order_id = info.get("broker_order_id")
        ts = info.get("timestamp")
        correlation_id = info.get("correlation_id")
        strategy_name = info.get("strategy")

        if event in ("fill", "partial_fill"):
            try:
                qty = float(info.get("fill_qty", 0.0) or 0.0)
                price = float(info.get("fill_price", 0.0) or 0.0)
            except Exception:
                self.logger.warning(
                    "fill_data_parse_failed",
                    event=event,
                    symbol=symbol,
                    fill_qty=info.get("fill_qty"),
                    fill_price=info.get("fill_price"),
                    exc_info=True,
                )
                qty = 0.0
                price = 0.0
            if qty <= 0 or price <= 0 or not symbol or not side:
                return
            self.on_fill(
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "broker_order_id": broker_order_id or None,
                    "timestamp": ts,
                    "strategy": strategy_name or None,
                    "correlation_id": correlation_id or None,
                    "trade_event": event,
                }
            )

    # ── Streaming Microstructure Handlers ──

    async def on_quote(self, symbol: str, quote: Any) -> None:
        """Process a live quote from the gateway stream for streaming OFI updates."""
        if self.streaming_scanner is None:
            return
        try:
            await self.streaming_scanner.on_quote(symbol, quote)
        except Exception as e:
            self.logger.debug("Streaming quote processing error", symbol=symbol, error=str(e))

    async def on_trade_data(self, symbol: str, trade: Any) -> None:
        """Process a live trade from the gateway stream for streaming TFI updates."""
        if self.streaming_scanner is None:
            return
        try:
            await self.streaming_scanner.on_trade(symbol, trade)
        except Exception as e:
            self.logger.debug("Streaming trade processing error", symbol=symbol, error=str(e))

    def on_streaming_update(self, symbol: str, features: Dict[str, float]) -> None:
        """Hot-patch SymbolState.meta with fresh microstructure features from streaming data."""
        state = self.symbol_states.get(symbol)
        if state is None:
            return
        state.meta["ofi"] = features.get("ofi", state.meta.get("ofi", 0.0))
        state.meta["tfi"] = features.get("tfi", state.meta.get("tfi", 0.0))
        state.meta["high_low_spread"] = features.get("high_low_spread", state.meta.get("high_low_spread", 0.0))

    async def run_scan(self):
        """
        Triggers a scan and updates the active symbol list/strategies.
        """
        if not self.scanner:
            self.logger.warning("Scanner not initialized")
            return

        try:
            results = await self.scanner.scan(regime=self.market_state.regime, scan_time=self.market_state.time)
        except Exception as e:
            self.error_counts["scanner"] += 1

            self.logger.error(
                "Scanner scan failed; retaining existing watchlist",
                error_code=ErrorCode.SCANNER_SCAN_FAILED.value,
                regime=getattr(self.market_state.regime, "value", str(self.market_state.regime)),
                scan_time=str(self.market_state.time),
                error=str(e),
                run_id=self.run_id,
            )
            return

        # results is a ScanResult object
        self.apply_scan_result(results)

        # PRD 8.1: persist scanner snapshots for analytics.
        if self.db:
            scan_ts = results.generated_at
            scan_regime = results.regime.value

            def make_snapshot_writer(snapshot: ScannerSnapshot):
                def _write(session: Session) -> None:
                    session.add(snapshot)

                return _write

            for w in results.watchlist:
                # JSON columns should be serializable; convert datetimes via str().
                features_json = self._sanitize_features_snapshot(asdict(w.features))
                snapshot = ScannerSnapshot(
                    timestamp=scan_ts,
                    regime=scan_regime,
                    symbol=w.symbol,
                    scanner_score=float(w.score),
                    strategies_json={"strategies": list(w.strategies)},
                    features_json=features_json,
                )
                self.db.write("scanner_snapshot", make_snapshot_writer(snapshot))

    def apply_scan_result(self, scan_result: ScanResult | List[WatchlistSymbol]):
        """
        Applies scan results with churn throttling.
        PRD 6.3: accepts a ScanResult; for backward compatibility may also accept
        a list of WatchlistSymbol objects.
        """
        scan_results = self._normalize_scan_results(scan_result)

        # Determinism: respect scanner-provided ordering for adds/retained updates and
        # use a stable ordering for removals.
        index_symbol = str(self.config.get("index_symbol", "SPY") or "")

        existing_non_index = set(self.symbol_states.keys())
        existing_non_index.discard(index_symbol)

        final_adds, final_removes = self._calculate_scan_churn(scan_results, existing_non_index, index_symbol)

        new_target_map = {res.symbol: res for res in scan_results}

        self._apply_scan_diff(final_adds, final_removes, new_target_map, scan_results, existing_non_index)

    def _normalize_scan_results(self, scan_result: ScanResult | List[WatchlistSymbol]) -> List[Any]:
        scan_results: List[Any]
        if isinstance(scan_result, ScanResult):
            scan_results = list(scan_result.watchlist)
        else:
            scan_results = list(scan_result)

        index_symbol = str(self.config.get("index_symbol", "SPY") or "")

        # PRD 1.1: keep total WS subscriptions within Alpaca's practical ~30 ticker limit.
        has_index_in_scan = any(getattr(res, "symbol", None) == index_symbol for res in scan_results)
        reserve_for_index = 0 if has_index_in_scan else (1 if index_symbol else 0)
        hard_cap = max(0, 30 - reserve_for_index)
        if len(scan_results) > hard_cap:
            self.logger.warning(
                "Clamping scan results to Alpaca WS limit",
                requested=len(scan_results),
                clamped=hard_cap,
                reserve_for_index=reserve_for_index,
                index_symbol=index_symbol or None,
            )
            scan_results = list(scan_results)[:hard_cap]
        return scan_results

    def _calculate_scan_churn(
        self,
        scan_results: List[Any],
        existing_non_index: Set[str],
        index_symbol: str,
    ) -> Tuple[List[str], List[str]]:
        desired_symbols = [res.symbol for res in scan_results]
        desired_set = set(desired_symbols)

        # 1. Identify Candidates for Add/Remove (deterministic ordering)
        to_add = [s for s in desired_symbols if s not in existing_non_index and s != index_symbol]
        to_remove = sorted([s for s in existing_non_index if s not in desired_set])

        # 2/3. Throttle churn as a single total budget per scan interval (deterministic).
        # Prefer removals first to keep subscriptions bounded, then add new names.
        budget = int(self.max_churn_per_scan)
        final_removes = to_remove[: max(0, min(len(to_remove), budget))]
        budget -= len(final_removes)
        final_adds = to_add[: max(0, min(len(to_add), budget))]

        self.logger.info(
            "Applying Scan Churn",
            requested_add=len(to_add),
            actual_add=len(final_adds),
            requested_remove=len(to_remove),
            actual_remove=len(final_removes),
        )
        return final_adds, final_removes

    def _apply_scan_diff(
        self,
        final_adds: List[str],
        final_removes: List[str],
        new_target_map: Dict[str, Any],
        scan_results: List[Any],
        existing_non_index: Set[str],
    ) -> None:
        self._process_scan_removals(final_removes)
        self._process_scan_additions(final_adds, new_target_map)
        self._update_retained_strategies(scan_results, existing_non_index, new_target_map)

    def _process_scan_removals(self, final_removes: List[str]) -> None:
        """Remove symbols from active trading set."""
        for sym in final_removes:
            if sym not in self.symbol_states:
                continue
            if self.symbol_states[sym].position:
                self.logger.info("Retaining symbol with position outside scan", symbol=sym)
                continue

            # Best-effort: cancel open orders before removing
            self._cleanup_orders_for_symbol(sym)

            self.logger.info("Dropping symbol from active set", symbol=sym)
            del self.symbol_states[sym]
            if self.on_subscription_change:
                self.on_subscription_change("remove", sym)
            if self.streaming_scanner:
                self.streaming_scanner.remove_symbol(sym)

    def _cleanup_orders_for_symbol(self, sym: str) -> None:
        """Cancel any open orders for a symbol being removed."""
        if not (self.db and self.order_executor):
            return

        try:
            # Cancel via broker API
            try:
                self.order_executor.cancel_all_for_symbol(sym)
            except Exception:
                self.logger.debug("Cancel via broker API failed", symbol=sym, exc_info=True)

            # Also cancel by looking up order IDs in DB
            broker_ids = self._get_pending_order_ids(sym)
            if broker_ids:
                self.logger.info(
                    "Clean-up canceling open orders for removed symbol",
                    symbol=sym,
                    count=len(broker_ids),
                )
                for oid in sorted(set(broker_ids)):
                    self.order_executor.cancel_by_broker_order_id(oid)
        except Exception as e:
            self.logger.warning("Clean-up cancel failed", symbol=sym, error=str(e))

    def _get_pending_order_ids(self, sym: str) -> List[str]:
        """Get pending order IDs for a symbol from database."""
        broker_ids: List[str] = []

        def _collect(session: Session, sym: str = sym) -> None:
            nonlocal broker_ids
            rows = (
                session.query(DbOrder)
                .filter(DbOrder.symbol == sym)
                .filter(DbOrder.broker_order_id.isnot(None))
                .order_by(DbOrder.id.desc())
                .limit(25)
                .all()
            )
            broker_ids = [str(r.broker_order_id) for r in rows if r.broker_order_id]

        assert self.db is not None
        self.db.write("collect_pending_cancel", _collect)
        return broker_ids

    def _process_scan_additions(self, final_adds: List[str], new_target_map: Dict[str, Any]) -> None:
        """Add new symbols from scan to active trading set."""
        for sym in final_adds:
            scanned_sym = new_target_map[sym]
            self.logger.info("Adding symbol to active set", symbol=sym)

            # Initialize SymbolState
            self.symbol_states[sym] = SymbolState(
                symbol=sym,
                bars=deque(maxlen=500),
                position=None,
                indicators={},
                open_orders={},
                allowed_strategies=list(scanned_sym.strategies),
                meta={},
            )

            # Build and apply meta from scan features
            meta = self._build_scan_meta(scanned_sym)
            self.symbol_states[sym].meta.update(meta)

            if self.on_subscription_change:
                self.on_subscription_change("add", sym)
            if self.streaming_scanner:
                self.streaming_scanner.add_symbol(sym)

    def _build_scan_meta(self, scanned_sym: Any) -> Dict[str, Any]:
        """Build symbol metadata from scan result."""
        features = scanned_sym.features
        features_snapshot = self._sanitize_features_snapshot(asdict(features)) if features else {}

        meta: Dict[str, Any] = {
            "added_at": str(self.market_state.time),
            "initial_score": float(scanned_sym.score),
            "sector": "unknown",
            "earnings_date": None,
            "iv_rank": 0.0,
            "short_interest": 0.0,
            "call_put_ratio": 0.0,
            "flow_bias": 0.0,
            "dof_score": 0.0,
            "relative_strength": 0.0,
            "atr": 0.0,
            "flow_zscore": 0.0,
            "gap_pct": 0.0,
            "premarket_volume": 0.0,
            "features_snapshot": features_snapshot,
            "features": features,  # Store raw object for StrategyEngine injection
        }

        if features:
            self._enrich_meta_from_features(meta, features)

        return meta

    def _determine_flow_bias(self, features: Any) -> str:
        """Determine flow bias from features."""
        if not features or not features.extra:
            return "neutral"
        prem = float(features.extra.get("flow_total_premium") or 0.0)
        if prem > 1000000:
            return "bullish"
        elif prem < -1000000:
            return "bearish"
        return "neutral"

    def _enrich_meta_from_features(self, meta: Dict[str, Any], features: Any) -> None:
        """Enrich metadata with feature values."""
        if features.extra:
            meta["sector"] = str(features.extra.get("sector") or "unknown")
            earnings = features.extra.get("earnings_date")
            meta["earnings_date"] = str(earnings) if earnings else None
            meta["iv_rank"] = float(features.extra.get("iv_rank") or 0.0)
            meta["short_interest"] = float(features.extra.get("short_interest") or 0.0)
            # Fetch ATR from extra if available (legacy fallback)
            if "atr" in features.extra and math.isclose(float(meta.get("atr") or 0.0), 0.0, abs_tol=1e-9):
                meta["atr"] = float(features.extra.get("atr") or 0.0)

        meta["call_put_ratio"] = features.call_put_ratio
        meta["premarket_volume"] = features.premarket_volume
        meta["flow_zscore"] = features.flow_zscore
        meta["gap_pct"] = features.gap_pct

        # New Alpha Signals (PRD Phase 2)
        meta["relative_strength"] = float(getattr(features, "relative_strength", 0.0))
        meta["flow_bias"] = float(getattr(features, "flow_bias", 0.0))
        meta["dof_score"] = float(getattr(features, "dof_score", 0.0))
        meta["orb_high"] = float(getattr(features, "orb_high", 0.0))
        meta["orb_low"] = float(getattr(features, "orb_low", 0.0))
        meta["alpha_score"] = float(getattr(features, "alpha_score", 0.0))
        meta["alpha_rank"] = int(getattr(features, "alpha_rank", 0))
        meta["prior_day_high"] = float(getattr(features, "prior_day_high", 0.0))
        meta["prior_day_low"] = float(getattr(features, "prior_day_low", 0.0))
        # ATR might be a top-level field or in extra
        if hasattr(features, "atr") and features.atr is not None:
            meta["atr"] = float(features.atr)
        elif "atr" in (features.extra or {}):
            meta["atr"] = float(features.extra.get("atr") or 0.0)

        # Cointegrated Pair Trading Metadata
        if features.extra and "pair_id" in features.extra:
            meta["pair_trade"] = {
                "pair_id": features.extra.get("pair_id"),
                "side": features.extra.get("pair_side"),
                "partner": features.extra.get("pair_partner"),
                "partner_price": features.extra.get("pair_partner_price"),
                "hedge_ratio": features.extra.get("hedge_ratio"),
                "z_score": features.extra.get("spread_zscore"),
            }

        # Forward Atlas factor scores from SymbolFeatures.extra → meta
        if features.extra:
            for key, val in features.extra.items():
                if key.startswith("atlas_"):
                    meta[key] = val
            # Map composite to ml_prediction for MLDirectionalStrategy
            if "atlas_composite" in features.extra:
                meta["ml_prediction"] = features.extra["atlas_composite"]

    def _update_retained_strategies(
        self,
        scan_results: List[Any],
        existing_non_index: Set[str],
        new_target_map: Dict[str, Any],
    ) -> None:
        desired_symbols = [res.symbol for res in scan_results]
        for sym in [s for s in desired_symbols if s in existing_non_index]:
            scanned_sym = new_target_map[sym]
            if sym in self.symbol_states:
                self.symbol_states[sym].allowed_strategies = list(scanned_sym.strategies)
                if scanned_sym.features:
                    features = scanned_sym.features
                    # Update metadata for retained symbols
                    self.symbol_states[sym].meta["flow_bias"] = float(getattr(features, "flow_bias", 0.0))
                    self.symbol_states[sym].meta["dof_score"] = float(getattr(features, "dof_score", 0.0))
                    self.symbol_states[sym].meta["relative_strength"] = float(
                        getattr(features, "relative_strength", 0.0)
                    )
                    self.symbol_states[sym].meta["orb_high"] = float(getattr(features, "orb_high", 0.0))
                    self.symbol_states[sym].meta["orb_low"] = float(getattr(features, "orb_low", 0.0))
                    self.symbol_states[sym].meta["premarket_volume"] = features.premarket_volume
                    # Update ATR
                    if hasattr(features, "atr") and features.atr is not None:
                        self.symbol_states[sym].meta["atr"] = float(features.atr)
                    elif "atr" in (features.extra or {}):
                        self.symbol_states[sym].meta["atr"] = float(features.extra.get("atr") or 0.0)

                    self.symbol_states[sym].meta["features_snapshot"] = self._sanitize_features_snapshot(
                        asdict(features)
                    )

                    # Forward Atlas factor scores for retained symbols
                    if features.extra:
                        for key, val in features.extra.items():
                            if key.startswith("atlas_"):
                                self.symbol_states[sym].meta[key] = val
                        if "atlas_composite" in features.extra:
                            self.symbol_states[sym].meta["ml_prediction"] = features.extra["atlas_composite"]

    async def reconcile_broker_state(self) -> None:
        """
        Reloads open positions and open orders from Alpaca and reconciles local state.
        """
        result = await self._fetch_broker_data()
        if not result:
            return

        _, positions, orders, closed_orders = result

        broker_order_ids: set[str] = set()
        broker_position_symbols: set[str] = set()

        # Build set of symbols with Cerberus-owned orders to filter positions
        own_order_symbols: set[str] = set()
        for o_list in (orders or [], closed_orders or []):
            for o in o_list if isinstance(o_list, list) else []:
                coid = str(getattr(o, "client_order_id", "") or "")
                if coid.startswith("cerberus_"):
                    sym = str(getattr(o, "symbol", "") or "")
                    if sym:
                        own_order_symbols.add(sym)

        # Also include symbols we're already tracking locally
        own_order_symbols.update(self.symbol_states.keys())

        # Filter positions to only those belonging to Cerberus
        if positions:
            positions = [p for p in positions if str(getattr(p, "symbol", "")) in own_order_symbols]

        # Reconcile Positions
        self._reconcile_positions(positions, broker_position_symbols)

        # Reconcile Orders
        self._reconcile_orders(orders, closed_orders, broker_order_ids)

        self.logger.info(
            "Reconciliation complete",
            positions=len(positions or []),
            open_orders=len(orders or []),
            closed_orders=len(closed_orders or []),
            symbols_tracked=len(self.symbol_states),
        )

    async def _fetch_broker_data(self) -> Optional[tuple[Any, Any, Any, Any]]:
        if self.broker_client is None:
            return None

        trading_client = self.broker_client.trading_client
        import asyncio

        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        def _get_orders_list(status: QueryOrderStatus, limit: Optional[int] = None) -> List[Any]:
            req = GetOrdersRequest(status=status, limit=limit)
            resp = trading_client.get_orders(req)
            return resp if isinstance(resp, list) else []

        try:
            account = await asyncio.to_thread(trading_client.get_account)  # type: ignore
            positions = await asyncio.to_thread(trading_client.get_all_positions)
            orders = await asyncio.to_thread(_get_orders_list, QueryOrderStatus.OPEN)
        except Exception as e:
            self.error_counts["orders"] += 1

            self.logger.error(
                "Reconcile fetch failed",
                error_code=ErrorCode.RECONCILE_FETCH_FAILED.value,
                error=str(e),
            )
            return None

        # Best-effort: also fetch recently closed orders
        closed_orders: List[Any] = []
        try:
            closed_orders = await asyncio.to_thread(_get_orders_list, QueryOrderStatus.CLOSED, 200)
        except Exception:
            closed_orders = []

        return account, positions, orders, closed_orders

    def _reconcile_positions(self, positions: Any, broker_position_symbols: set[str]) -> None:
        """Reconcile local positions with broker state."""
        if not positions:
            positions = []

        for p in positions:
            try:
                sym = str(getattr(p, "symbol", ""))
                if not sym:
                    continue
                broker_position_symbols.add(sym)

                state = self._get_or_create_symbol_state(sym)

                # Skip if recently updated or has pending orders
                if self._should_skip_reconcile(state, sym):
                    continue

                # Create reconciled position
                self._reconcile_single_position(state, p)
            except Exception as e:
                self.logger.warning("Position reconcile failed", error=str(e))

        # Check for local positions missing at broker
        self._handle_position_mismatch(broker_position_symbols)

    def _should_skip_reconcile(self, state: SymbolState, sym: str) -> bool:
        """Check if position reconciliation should be skipped."""
        # H1 fix: Skip if recently updated by fill
        if state.position:
            time_since_update = (datetime.now(timezone.utc) - state.position.last_updated).total_seconds()
            if time_since_update < 2.0:
                self.logger.debug(
                    "Skipping position reconciliation, recently updated",
                    symbol=sym,
                    seconds_since_update=round(time_since_update, 2),
                )
                return True

        # M4 fix: Skip if pending orders exist
        if state.open_orders:
            self.logger.debug(
                "Skipping position reconciliation, pending orders exist",
                symbol=sym,
                pending_order_count=len(state.open_orders),
            )
            return True

        return False

    def _reconcile_single_position(self, state: SymbolState, broker_pos: Any) -> None:
        """Reconcile a single position from broker data."""
        sym = str(getattr(broker_pos, "symbol", ""))
        side_raw = getattr(broker_pos, "side", "")
        side = Side.LONG if str(side_raw).lower() == "long" else Side.SHORT
        qty = float(getattr(broker_pos, "qty", getattr(broker_pos, "quantity", 0.0)) or 0.0)
        avg_price = float(getattr(broker_pos, "avg_entry_price", getattr(broker_pos, "avg_price", 0.0)) or 0.0)

        # Preserve existing position data where available
        existing = state.position
        if existing is not None and existing.side == side:
            # Same side — update qty/price from broker, preserve all exit management state
            existing.qty = qty
            existing.avg_price = avg_price
            existing.unrealized_pnl = 0.0
            existing.last_updated = datetime.now(timezone.utc)
            state.position = existing
        else:
            # New position or side flip — create fresh (no prior state to preserve)
            state.position = Position(
                symbol=sym,
                side=side,
                qty=qty,
                avg_price=avg_price,
                unrealized_pnl=0.0,
                realized_pnl=getattr(existing, "realized_pnl", 0.0) if existing else 0.0,
                strategy=existing.strategy if existing else "unknown",
                entry_time=getattr(existing, "entry_time", None) if existing else None,
                correlation_id=getattr(existing, "correlation_id", "") if existing else "",
                regime_tags_at_entry=getattr(existing, "regime_tags_at_entry", {}) if existing else {},
            )

    def _handle_position_mismatch(self, broker_position_symbols: set[str]) -> None:
        """Handle local positions missing at broker."""
        missing = [
            sym
            for sym, st in self.symbol_states.items()
            if st.position is not None and sym not in broker_position_symbols
        ]

        if not missing:
            return

        mode = str(self.config.get("position_mismatch_mode", "halt")).lower()
        self.logger.error(
            "Position mismatch detected (local position missing at broker)",
            symbols=sorted(missing),
            mode=mode,
        )

        # Clear stale local positions to prevent PositionManager from
        # submitting exit orders against non-existent broker positions
        for sym in missing:
            st = self.symbol_states.get(sym)
            if st is not None:
                self.logger.warning(
                    "Clearing stale local position (missing at broker)",
                    symbol=sym,
                    side=getattr(st.position, "side", None),
                    qty=getattr(st.position, "qty", None),
                )
                st.position = None

        if mode in ("halt", "stop"):
            self._set_risk_mode("off")

    def _clear_all_features(self) -> None:
        """
        M1 fix: Clear all cached indicators across all symbol states.

        Called on:
        - Market regime changes (to prevent stale regime-sensitive indicators)
        - Daily boundaries (safety net)
        """
        for state in self.symbol_states.values():
            state.indicators.clear()

        self.logger.debug(
            "Cleared feature cache",
            symbols_cleared=len(self.symbol_states),
        )

    def _reconcile_orders(self, orders: Any, closed_orders: Any, broker_order_ids: set[str]) -> None:
        """Reconcile local order state with broker orders."""
        if not orders:
            orders = []

        # Sync open orders to local state
        self._sync_open_orders_to_state(orders, broker_order_ids)

        # Cancel stale orders
        self._cancel_stale_orders(orders)

        # DB reconciliation
        if self.db:
            self._reconcile_db_orders(orders, closed_orders, broker_order_ids)

        # Synthesize missed fills
        if self.db and closed_orders:
            self._synthesize_missed_fills(closed_orders)

    def _sync_open_orders_to_state(self, orders: List[Any], broker_order_ids: set[str]) -> None:
        """Sync broker open orders into symbol state."""
        for o in orders:
            try:
                sym = str(getattr(o, "symbol", ""))
                oid = str(getattr(o, "id", ""))
                broker_order_ids.add(oid)
                if not sym or not oid:
                    continue

                state = self._get_or_create_symbol_state(sym)
                state.open_orders[oid] = {
                    "status": str(getattr(o, "status", "")),
                    "qty": float(getattr(o, "qty", getattr(o, "quantity", 0.0)) or 0.0),
                    "side": str(getattr(o, "side", "")),
                    "limit_price": getattr(o, "limit_price", None),
                    "created_at": str(getattr(o, "created_at", "") or ""),
                }
            except Exception as e:
                self.logger.warning("Order reconcile failed", error=str(e))

    def _cancel_stale_orders(self, orders: List[Any]) -> None:
        """Cancel orders older than configured max age."""
        max_age_sec = int(self.config.get("max_open_order_age_sec", 0))
        if max_age_sec <= 0 or not self.order_executor:
            return

        now = self.clock()
        for o in orders:
            try:
                oid = str(getattr(o, "id", "") or "")
                created_at = getattr(o, "created_at", None)
                if not oid or created_at is None:
                    continue

                age = (now - created_at).total_seconds()
                if age >= max_age_sec:
                    self.logger.warning(
                        "Cancelling stale open order",
                        broker_order_id=oid,
                        age_sec=float(age),
                        max_age_sec=max_age_sec,
                    )
                    self.order_executor.cancel_by_broker_order_id(oid)
            except Exception as e:
                self.logger.warning("Stale-order cancel failed", error=str(e))

    def _reconcile_db_orders(self, orders: Any, closed_orders: Any, broker_order_ids: set[str]) -> None:
        """Reconcile DB order records with broker state."""
        if not self.db:
            return

        now = self.clock()

        def _reconcile_db(session: Session) -> None:
            # Update open orders
            self._update_open_order_statuses(session, orders or [], now)
            # Update closed orders
            self._update_closed_order_statuses(session, closed_orders or [], now)
            # Mark stale orders as cancelled
            self._mark_stale_orders_cancelled(session, broker_order_ids, now)

        try:
            self.db.write("order_reconcile", _reconcile_db)
        except Exception as e:
            self.logger.warning("DB reconcile failed", error=str(e))

    def _update_open_order_statuses(self, session: Session, orders: List[Any], now: datetime) -> None:
        """Update DB statuses for orders still open at broker."""
        for o in orders:
            oid = str(getattr(o, "id", ""))
            status_val = str(getattr(o, "status", "") or "")
            if not oid:
                continue

            row = session.query(DbOrder).filter(DbOrder.broker_order_id == oid).order_by(DbOrder.id.desc()).first()
            if row:
                self._apply_reconcile_status(row, status_val, now)

    def _update_closed_order_statuses(self, session: Session, closed_orders: List[Any], now: datetime) -> None:
        """Update DB statuses for recently closed orders."""
        for o in closed_orders:
            oid = str(getattr(o, "id", "") or "")
            status_val = str(getattr(o, "status", "") or "")
            if not oid or not status_val:
                continue

            row = session.query(DbOrder).filter(DbOrder.broker_order_id == oid).order_by(DbOrder.id.desc()).first()
            if row:
                self._apply_reconcile_status(row, status_val, now)

    def _mark_stale_orders_cancelled(self, session: Session, broker_order_ids: set[str], now: datetime) -> None:
        """Mark DB orders as cancelled if broker no longer has them."""
        stale = (
            session.query(DbOrder)
            .filter(DbOrder.status.notin_(["filled", "canceled", "cancelled"]))
            .filter(DbOrder.broker_order_id.isnot(None))
            .all()
        )
        for row in stale:
            if row.broker_order_id and row.broker_order_id not in broker_order_ids:
                row.status = "cancelled"
                row.time_last_update = now
                meta = row.meta_json or {}
                meta.update(
                    {
                        "reconciled": True,
                        "reason": "broker_missing",
                        "last_reconciled": str(now),
                    }
                )
                row.meta_json = meta

    def _apply_reconcile_status(self, row: DbOrder, status_val: str, now: datetime) -> None:
        """Apply reconciliation status update to an order row."""
        row.status = status_val
        row.time_last_update = now
        meta = row.meta_json or {}
        meta.update(
            {
                "reconciled": True,
                "broker_status": status_val,
                "last_reconciled": str(now),
            }
        )
        row.meta_json = meta

    def _synthesize_missed_fills(self, closed_orders: Any) -> None:
        now = self.clock()

        for o in closed_orders:
            self._handle_closed_order_reconciliation(o, now)

    def _handle_closed_order_reconciliation(self, o: Any, now: datetime) -> None:
        broker_id = str(getattr(o, "id", "") or "")
        sym = str(getattr(o, "symbol", "") or "")
        if not broker_id or not sym:
            return

        filled_qty = float(getattr(o, "filled_qty", 0.0) or 0.0)
        filled_avg_price = float(getattr(o, "filled_avg_price", 0.0) or 0.0)
        if filled_qty <= 0 or filled_avg_price <= 0:
            return

        side_raw = getattr(o, "side", None)
        side_val = getattr(side_raw, "value", None) or str(side_raw or "")
        if not side_val:
            return

        fill_ts = getattr(o, "filled_at", None) or getattr(o, "updated_at", None) or now

        corr = str(getattr(o, "client_order_id", "") or "")

        order_id: Optional[int] = None
        if self.db:
            order_id = self._get_reconciled_order_id(broker_id, corr, sym, side_val, filled_qty, o, fill_ts)

        if order_id is None:
            return

        self.logger.warning(
            "Synthesizing fill from closed order",
            symbol=sym,
            qty=filled_qty,
            price=filled_avg_price,
            order_id=broker_id,
        )

        if self.db:
            self._persist_synthesized_fill(order_id, fill_ts, filled_avg_price, filled_qty)

    def _get_reconciled_order_id(
        self,
        broker_id: str,
        corr: str,
        sym: str,
        side_val: str,
        filled_qty: float,
        o: Any,
        fill_ts: datetime,
    ) -> Optional[int]:
        order_id: Optional[int] = None

        def _ensure_order_and_check(session: Session) -> None:
            nonlocal order_id
            row = (
                session.query(DbOrder).filter(DbOrder.broker_order_id == broker_id).order_by(DbOrder.id.desc()).first()
            )
            if row is None:
                session.add(
                    DbOrder(
                        correlation_id=corr,
                        symbol=sym,
                        side=side_val,
                        qty=filled_qty,
                        type="unknown",
                        limit_price=None,
                        status=str(getattr(o, "status", "") or "filled"),
                        time_placed=getattr(o, "created_at", None) or fill_ts,
                        time_last_update=fill_ts,
                        broker_order_id=broker_id,
                        meta_json={"recovered_from_reconcile": True},
                    )
                )
                session.flush()
                row = (
                    session.query(DbOrder)
                    .filter(DbOrder.broker_order_id == broker_id)
                    .order_by(DbOrder.id.desc())
                    .first()
                )

            if row is None or row.id is None:
                return

            # If any fill exists, don't synthesize another aggregate fill.
            existing = session.query(DbFill).filter(DbFill.order_id == int(row.id)).order_by(DbFill.id.desc()).first()
            if existing is None:
                order_id = int(row.id)

        if self.db:
            self.db.write("ensure_order_reconcile", _ensure_order_and_check)
        return order_id

    def _persist_synthesized_fill(
        self,
        order_id: int,
        fill_ts: datetime,
        filled_avg_price: float,
        filled_qty: float,
    ) -> None:
        fill_ts = self._normalize_timestamp(fill_ts)

        def _write_fill(session: Session) -> None:
            session.add(
                DbFill(
                    order_id=order_id,
                    fill_time=fill_ts,
                    fill_price=filled_avg_price,
                    fill_qty=filled_qty,
                )
            )

        if self.db:
            self.db.write("synthesize_fill", _write_fill)

    async def reconcile_loop(self):
        """
        Periodic reconciliation of broker state.
        """
        import asyncio

        interval = int(self.config.get("reconcile_interval_sec", 600))
        while True:
            await asyncio.sleep(max(1, interval))
            try:
                await self.reconcile_broker_state()
            except Exception as e:
                self.logger.error(
                    "reconcile_loop_iteration_failed",
                    error=str(e),
                    exc_info=True,
                )
