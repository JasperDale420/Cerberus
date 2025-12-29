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
from src.core.domain import Position, ScanResult, Side, WatchlistSymbol
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.engine.health import HealthMonitor
from src.engine.market import MarketStateManager
from src.engine.orders import OrderExecutor
from src.engine.position_manager import PositionManager
from src.engine.risk import RiskManager
from src.engine.strategy_engine import StrategyEngine, StrategyRouting
from src.scanner.core import Scanner
from src.strategies.base import BaseStrategy, Signal, SymbolState


class ExecutionEngine:
    """
    Orchestrates data flow, strategy execution, and order management.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: StructuredLogger,
        db: Optional[DatabaseDatabase] = None,
        alpaca_client: Optional[AlpacaClient] = None,
        run_id: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.config = config
        self.logger = logger
        self.db = db
        self.alpaca_client = alpaca_client
        self.run_id = run_id
        self.clock: Callable[[], datetime] = clock or (
            lambda: datetime.now(timezone.utc)
        )

        self.risk_manager = RiskManager(config, logger)
        self.order_executor = (
            OrderExecutor(alpaca_client, logger, db, clock=self.clock)
            if alpaca_client
            else None
        )
        self.scanner: Optional[Scanner] = None  # Injected later or via init

        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_engine: Optional[StrategyEngine] = None
        self.position_manager = PositionManager()
        self.symbol_states: Dict[str, SymbolState] = {}
        # In-memory trade capture for backtests and offline analysis (best-effort).
        self.closed_trades: List[Any] = []

        # Extracted Collaborators
        self.health = HealthMonitor(config, logger, run_id, self.clock)
        self.market_manager = MarketStateManager(
            config, logger, db, self.clock, on_error=self._inc_error
        )

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
            from src.core.indicators import (
                RollingEMA,
                RollingRSI,
                RollingSMA,
                RollingStd,
            )

            if not isinstance(state.indicators, dict):
                return

            strategies = list(getattr(state, "allowed_strategies", []) or [])
            # Determine required windows from config and enabled strategies.
            strat_cfgs = (
                self.config.get("strategies")
                if isinstance(self.config.get("strategies"), dict)
                else {}
            )

            ema_periods: set[int] = set()
            rsi_periods: set[int] = set()
            vol_sma_periods: set[int] = set()
            bb_periods: set[int] = set()

            for s in strategies:
                cfg = strat_cfgs.get(s) if isinstance(strat_cfgs, dict) else None
                if not isinstance(cfg, dict):
                    cfg = {}
                if s in ("trend_pullback", "vwap_trend_rider"):
                    ema_periods.add(int(cfg.get("ema_fast", 20)))
                    ema_periods.add(int(cfg.get("ema_slow", 50)))
                if s == "trend_pullback":
                    rsi_periods.add(int(cfg.get("rsi_len", 2)))
                if s in ("flow_momentum", "vwap_trend_rider"):
                    vol_sma_periods.add(20)
                if s == "index_mean_reversion":
                    bb_periods.add(int(cfg.get("bb_len", 20)))

            close = float(getattr(bar, "close", 0.0) or 0.0)
            volume = float(getattr(bar, "volume", 0.0) or 0.0)

            for p in sorted({x for x in ema_periods if x > 0}):
                key = f"_ema_close_obj:{p}"
                obj = state.indicators.get(key)
                if not isinstance(obj, RollingEMA):
                    obj = RollingEMA.from_period(p)
                    state.indicators[key] = obj
                prev = obj.value
                val = obj.update(close)
                state.indicators[f"ema_close:{p}:prev"] = prev
                state.indicators[f"ema_close:{p}"] = val

            for p in sorted({x for x in rsi_periods if x > 0}):
                key = f"_rsi_obj:{p}"
                obj = state.indicators.get(key)
                if not isinstance(obj, RollingRSI):
                    obj = RollingRSI(period=p)
                    state.indicators[key] = obj
                prev = obj.value
                rsi_val = obj.update(close)
                state.indicators[f"rsi:{p}:prev"] = prev
                state.indicators[f"rsi:{p}"] = rsi_val

            for p in sorted({x for x in vol_sma_periods if x > 0}):
                key = f"_sma_vol_obj:{p}"
                obj = state.indicators.get(key)
                if not isinstance(obj, RollingSMA):
                    obj = RollingSMA.create(p)
                    state.indicators[key] = obj
                prev = obj.value
                val = obj.update(volume)
                state.indicators[f"sma_vol:{p}:prev"] = prev
                state.indicators[f"sma_vol:{p}"] = val

            for p in sorted({x for x in bb_periods if x > 0}):
                key = f"_std_close_obj:{p}"
                obj = state.indicators.get(key)
                if not isinstance(obj, RollingStd):
                    obj = RollingStd.create(p)
                    state.indicators[key] = obj
                mean, std = obj.update(close)
                state.indicators[f"bb_mean:{p}"] = mean
                state.indicators[f"bb_std:{p}"] = std
        except Exception:
            # Best-effort: do not break trading.
            self._inc_error("execution")

    def flatten_all(self, *, reason: str = "eod") -> None:
        """
        Emergency flatten: close all positions and cancel all orders.

        Critical safety function used at end-of-day or when risk limits are breached.
        Executes with best-effort semantics to aggressively flatten the entire book.

        Workflow:
        1. Cancel all open orders via Alpaca
        2. Close all open positions via Alpaca
        3. Reset local position and watchlist state
        4. Log confirmation status (success/failure)

        Args:
            reason: Reason for flattening ("eod", "risk_breach", "manual", etc.)

        Returns:
            None

        Raises:
            Exception: If position_mismatch_mode is 'halt' and confirmation fails

        Side Effects:
            - Sends cancel_all_orders request to Alpaca
            - Sends close_all_positions request to Alpaca
            - Clears self.watchlist and all position state
            - Logs ERROR with FLATTEN_CONFIRMATION_FAILED if positions remain
            - May halt trading if confirmation fails (based on config)

        Note:
            - Best-effort operation; individual cancels/closes may fail
            - PRD 11.2: Includes confirmation check for position_mismatch_mode
            - Always executed at market close (16:00 ET) if flat_on_close enabled
            - Does not stop execution engine; use stop() for full shutdown
        """
        if not self.alpaca_client:
            self.logger.warning("Flatten requested but no Alpaca client", reason=reason)
            return

        # Stop taking new risk immediately.
        self._set_risk_mode("off")

        mismatch_mode = str(self.config.get("position_mismatch_mode", "halt")).lower()

        self.logger.info("Flatten starting", reason=reason, mismatch_mode=mismatch_mode)

        # 1) Cancel all open orders (best-effort).
        try:
            self.alpaca_client.trading_client.cancel_orders()
        except Exception as e:
            self.logger.error(
                "Cancel orders failed", reason=reason, error=str(e), exc_info=True
            )

        # 2) Close all open positions (best-effort; Alpaca can also cancel orders).
        try:
            self.alpaca_client.trading_client.close_all_positions(cancel_orders=True)
        except Exception as e:
            self.logger.error(
                "Close all positions failed",
                reason=reason,
                error=str(e),
                exc_info=True,
            )
            if mismatch_mode in ("halt", "stop", "raise"):
                raise

        # 3) Confirm flat state at broker (best-effort).
        open_positions = []
        open_orders = []
        positions_confirmed = False
        orders_confirmed = False
        try:
            open_positions = list(self.alpaca_client.trading_client.get_all_positions())
            positions_confirmed = True
        except Exception as e:
            self.logger.warning(
                "Failed to fetch broker positions after flatten", error=str(e)
            )
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            resp = self.alpaca_client.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            open_orders = resp if isinstance(resp, list) else []
            orders_confirmed = True
        except Exception as e:
            self.logger.warning(
                "Failed to fetch broker orders after flatten", error=str(e)
            )

        # 4) Best-effort local state reset; reconciliation loop will repair any mismatches.
        for state in self.symbol_states.values():
            state.position = None
            state.open_orders = {}

        if mismatch_mode in ("halt", "stop", "raise") and (
            not positions_confirmed or not orders_confirmed
        ):
            self.logger.error(
                "Flatten confirmation failed",
                positions_confirmed=positions_confirmed,
                orders_confirmed=orders_confirmed,
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
                raise RuntimeError(
                    "Flatten incomplete: broker still reports positions/orders"
                )
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
        self.max_churn_per_scan = config.get(
            "max_churn_per_scan", self.max_churn_per_scan
        )
        self.health.update_config(config)
        # RiskManager already has raw_config; update individual fields instead
        risk_cfg = (
            config.get("risk") if isinstance(config.get("risk"), dict) else None
        ) or {}
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
        self._set_risk_mode(
            str(
                risk_cfg.get(
                    "risk_mode", config.get("risk_mode", self.risk_manager.risk_mode)
                )
            )
        )
        self.risk_manager.max_notional_per_symbol = float(
            risk_cfg.get(
                "max_notional_per_symbol",
                config.get(
                    "max_notional_per_symbol", self.risk_manager.max_notional_per_symbol
                ),
            )
        )
        self._refresh_strategy_engine()

    def _refresh_strategy_engine(self) -> None:
        cfg = self.config or {}
        routing_cfg = cfg.get("strategy_routing") or {}

        strategies_cfg = (
            cfg.get("strategies") if isinstance(cfg.get("strategies"), dict) else {}
        )

        def _enabled(name: str, regime_key: str) -> bool:
            s_cfg = (
                strategies_cfg.get(name) if isinstance(strategies_cfg, dict) else None
            )
            if not isinstance(s_cfg, dict):
                return True
            enabled = bool(s_cfg.get("enabled", True))
            regimes = s_cfg.get("regimes")
            if isinstance(regimes, dict):
                r_cfg = regimes.get(regime_key)
                if isinstance(r_cfg, dict) and "enabled" in r_cfg:
                    enabled = bool(r_cfg.get("enabled"))
            return enabled

        def _names(regime_key: str) -> List[str]:
            raw = routing_cfg.get(regime_key, [])
            if isinstance(raw, list) and raw:
                names = [str(x) for x in raw]
            else:
                # Default: allow all registered strategies when routing not configured.
                names = sorted(self.strategies.keys())

            out: List[str] = []
            for n in names:
                if n not in self.strategies:
                    continue
                if not _enabled(n, regime_key):
                    continue
                out.append(n)
            return out

        routing = StrategyRouting(
            strategies_by_regime={
                Regime.BULL: _names("bull"),
                Regime.BEAR: _names("bear"),
                Regime.CHOP: _names("chop"),
            }
        )
        self.strategy_engine = StrategyEngine(
            self.strategies,
            routing,
            self.logger,
            on_error=lambda: self._inc_error("strategy"),
        )

    def on_bar(self, symbol_or_bar: Any, bar: Optional[Any] = None) -> None:
        """
        Process a new bar for a symbol.

        PRD 6.2: the WebSocket data handler should call `ExecutionEngine.on_bar(bar)`.
        Backward-compatible: callers may also call `on_bar(symbol, bar)`.
        """
        if bar is None:
            bar = symbol_or_bar
            symbol = getattr(bar, "symbol", None) or getattr(bar, "S", None)
            if not symbol:
                raise ValueError("on_bar(bar) requires bar.symbol")
            symbol = str(symbol)
        else:
            symbol = str(symbol_or_bar)

        import time as _time

        start_bar = _time.perf_counter()

        self.bars_processed += 1
        bar_time = getattr(bar, "time", getattr(bar, "timestamp", None))

        try:
            _bind = getattr(self.logger, "bind", None)
            log = (
                _bind(symbol=symbol, run_id=self.run_id)
                if callable(_bind)
                else self.logger
            )

            # Helper: Update State
            state = self._update_symbol_state(symbol, bar, bar_time)

            # Helper: Run Strategies
            self._run_strategies(symbol, bar, state, log)

            # Helper: Manage Positions
            self._manage_positions(symbol, state, log)

            # Emit periodic health metrics based on bar time (deterministic for replay).
            try:
                self._maybe_log_health(self.market_state.time)
            except Exception:
                self._inc_error("execution")

            # PRD 11.2: latency observability (best-effort; does not affect decisions).
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
            from src.core.errors import ErrorCode

            _bind = getattr(self.logger, "bind", None)
            log = (
                _bind(symbol=symbol, run_id=self.run_id)
                if callable(_bind)
                else self.logger
            )
            log.error(
                "Bar processing failed",
                error_code=ErrorCode.ENGINE_ON_BAR_FAILED.value,
                symbol=symbol,
                bar_time=str(bar_time or self.clock()),
                regime=getattr(
                    self.market_state.regime, "value", str(self.market_state.regime)
                ),
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
        # Update Market State (if symbol is index)
        if symbol == self.config.get("index_symbol", "SPY"):
            self.market_manager.update(bar, index_symbol=symbol)

        # Update Symbol State
        if symbol not in self.symbol_states:
            self.symbol_states[symbol] = SymbolState(
                symbol=symbol,
                bars=deque(maxlen=100),
                position=None,
                indicators={},
                open_orders={},
                allowed_strategies=[],
                meta={},
            )

        state = self.symbol_states[symbol]
        state.bars.append(bar)

        # PRD 7.2 VWAP Reversion: maintain intraday/session VWAP independent of bar deque window.
        try:
            bt = bar_time or self.clock()
            session_day = bt.date() if hasattr(bt, "date") else None
            prev_day = state.indicators.get("session_day")
            if session_day is not None and prev_day != session_day:
                state.indicators["session_day"] = session_day
                state.indicators["cum_pv"] = 0.0
                state.indicators["cum_vol"] = 0.0

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
            if cum_vol > 0:
                session_vwap = cum_pv / cum_vol
                state.indicators["session_vwap"] = session_vwap
                try:
                    bar.vwap = session_vwap
                except Exception:
                    pass
        except Exception:
            self._inc_error("execution")

        # PRD 11.2: update indicator cache once per bar per symbol.
        try:
            self._update_indicator_cache(state, bar)
        except Exception:
            self._inc_error("execution")

        # PRD 6.7: Update unrealized PnL deterministically on each bar.
        try:
            mark = float(getattr(bar, "close", 0.0) or 0.0)
            self.position_manager.update_unrealized_pnl(state, mark)
        except Exception:
            self._inc_error("execution")

        return state

    def _run_strategies(
        self, symbol: str, bar: Any, state: SymbolState, log: Any
    ) -> None:
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
                        regime=getattr(sig.regime, "value", str(sig.regime)),
                    )
                    if callable(s_bind)
                    else log
                )
                slog.info(
                    "Signal generated",
                    symbol=sig.symbol,
                    strategy=sig.strategy,
                    regime=getattr(sig.regime, "value", str(sig.regime)),
                    correlation_id=getattr(sig, "correlation_id", "") or None,
                    signal=sig,
                    run_id=self.run_id,
                )
                self.signals_generated += 1
                self._process_signal(sig)

    def _manage_positions(self, symbol: str, state: SymbolState, log: Any) -> None:
        # PositionManager exit checks (PRD 6.7 best-effort)
        try:
            decision = self.position_manager.on_bar(
                state,
                self.market_state,
                broker_managed_exits=bool(
                    getattr(self.order_executor, "broker_managed_exits", False)
                ),
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
            from src.core.errors import ErrorCode

            self.logger.error(
                "PositionManager exit failed",
                error_code=ErrorCode.POSITION_EXIT_FAILED.value,
                symbol=symbol,
                regime=getattr(
                    self.market_state.regime, "value", str(self.market_state.regime)
                ),
                error=str(e),
                exc_info=True,
            )

    def _process_signal(self, signal: Signal):
        """
        Passes signal to Risk Manager and then Order Executor.
        Persists signal to DB.
        """
        # PRD tracing: correlation_id is generated deterministically in Signal.__post_init__.

        _bind = getattr(self.logger, "bind", None)
        log = (
            _bind(
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=signal.correlation_id,
                run_id=self.run_id,
                regime=getattr(signal.regime, "value", str(signal.regime)),
            )
            if callable(_bind)
            else self.logger
        )
        log.info(
            "Processing signal",
            symbol=signal.symbol,
            strategy=signal.strategy,
            correlation_id=signal.correlation_id,
            regime=getattr(signal.regime, "value", str(signal.regime)),
            run_id=self.run_id,
        )

        # 1. Risk Check
        import time  # Import time here to avoid circular dependency or global import if not needed elsewhere

        # Gather active positions from symbol states
        active_positions = [
            s.position for s in self.symbol_states.values() if s.position is not None
        ]

        start_risk = time.perf_counter()
        try:
            intents = self.risk_manager.apply(
                signal,
                self.symbol_states[signal.symbol],
                self.market_state,
                current_positions=active_positions,
                account_equity=float(getattr(self.account, "equity", 0.0) or 0.0),
            )
        except Exception as e:
            self._inc_error("risk")
            from src.core.errors import ErrorCode

            log.error(
                "RiskManager.apply failed; dropping signal",
                error_code=ErrorCode.RISK_APPLY_FAILED.value,
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=getattr(signal, "correlation_id", "") or None,
                error=str(e),
                run_id=self.run_id,
            )
            return
        risk_latency = (time.perf_counter() - start_risk) * 1000

        # 2. Persist Signal
        if self.db:
            rejection_reason = None
            if not intents:
                rejection_reason = (
                    self.risk_manager.last_rejection_reason or "RISK_REJECTED"
                )
            ok = self.db.write(
                "signal",
                lambda session: session.add(
                    DbSignal(
                        correlation_id=signal.correlation_id,
                        symbol=signal.symbol,
                        strategy=signal.strategy,
                        regime=signal.regime.value,
                        time=signal.generated_at,
                        raw_side=signal.side.value,
                        raw_size=signal.size_hint,
                        accepted=bool(intents),
                        rejection_reason=rejection_reason,
                        meta_json=signal.meta,
                    )
                ),
            )
            if not ok:
                self._inc_error("db")

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

        # Stash deterministic entry context keyed by correlation_id for use on fills/trades.
        # Current behavior: only the first intent is an entry intent.
        intent0 = intents[0]
        state = self.symbol_states[signal.symbol]
        pending = state.meta.setdefault("pending_entries", {})
        if isinstance(pending, dict) and hasattr(intent0, "qty"):
            # Carry per-strategy config that affects trade lifecycle (e.g., max hold).
            strat_cfg = None
            if isinstance(self.config.get("strategies"), dict):
                strat_cfg = (self.config.get("strategies") or {}).get(signal.strategy)
            max_hold_minutes = None
            if isinstance(strat_cfg, dict):
                max_hold_minutes = strat_cfg.get("max_hold_minutes")

            features_payload = state.meta.get("features_snapshot")
            if isinstance(features_payload, dict):
                features_payload = self._sanitize_features_snapshot(features_payload)

            pending[signal.correlation_id] = {
                "strategy": signal.strategy,
                "regime": signal.regime.value,
                "entry_price": signal.entry_price,
                "stop_price": signal.stop_price,
                "target_price": signal.target_price,
                "qty": intent0.qty,
                "open_risk": abs(signal.entry_price - signal.stop_price)
                * float(intent0.qty),
                "features": (
                    {
                        **(features_payload or {}),
                        "scanner_score": state.meta.get("score"),
                    }
                    if isinstance(features_payload, dict)
                    else features_payload
                ),
                "entry_time": signal.generated_at,
                "max_hold_seconds": (
                    int(float(max_hold_minutes) * 60)
                    if max_hold_minutes is not None
                    else None
                ),
            }

        # 3. Order Execution
        self._execute_signal_intents(signal, intents, log, risk_latency)

    def _execute_signal_intents(
        self, signal: Signal, intents: List[Any], log: Any, risk_latency: float
    ) -> None:
        if self.order_executor:
            # PRD 11.4: configurable safe behavior on persistent DB failures.
            if self.db:
                mode = str((self.config.get("db_trading_mode") or "warn")).lower()
                if mode in ("halt", "stop"):
                    buf_len = 0
                    buf_max = 0
                    try:
                        buf_len = int(self.db.write_buffer_len())
                        buf_max = int(self.db.write_buffer_max())
                    except Exception:
                        pass

                    threshold = int(
                        self.config.get(
                            "db_trading_halt_buffer_len", max(1, buf_max // 2 or 1)
                        )
                    )
                    if self.db.last_db_write_error and buf_len >= threshold:
                        # Fail-safe: stop new orders.
                        self._set_risk_mode("off")
                        log.error(
                            "Halting trading due to persistent DB failures",
                            last_db_error=self.db.last_db_write_error,
                            db_buffer_len=buf_len,
                            db_buffer_max=buf_max,
                            threshold=threshold,
                        )
                        return

            import time

            for intent in intents:
                try:
                    start_exec = time.perf_counter()
                    self.order_executor.submit(intent)
                    exec_latency = (time.perf_counter() - start_exec) * 1000
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
        else:
            self.logger.warning(
                "No OrderExecutor available (paper/mock mode)",
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=signal.correlation_id,
                intents=intents,
                run_id=self.run_id,
            )

    def on_fill(self, fill: Dict[str, Any]):
        """
        Updates position state and persists Trades on close.
        fill struct: {
            symbol, side (buy/sell), qty, price,
            order_id/broker_order_id (optional), timestamp, strategy, correlation_id
        }
        """
        symbol = fill["symbol"]
        if symbol not in self.symbol_states:
            self.logger.warning("Fill received for unknown symbol", symbol=symbol)
            return

        state = self.symbol_states[symbol]
        fill_qty = float(fill["qty"])
        fill_price = float(fill["price"])
        fill_ts = fill.get("timestamp") or self.market_state.time
        # Avoid leaking str(None) == "None" into persisted correlation IDs.
        corr = str(fill.get("correlation_id") or "")

        # PRD 2.1/3.2: enforce non-empty correlation_id for persistence/tracing.
        corr_id = corr
        if not corr_id:
            broker_order_id = fill.get("broker_order_id")
            if broker_order_id is not None and str(broker_order_id):
                corr_id = f"alpaca-{str(broker_order_id)}"
            else:
                import hashlib

                seed = f"{symbol}|{fill.get('side', '')}|{fill_ts}"
                corr_id = (
                    "fill-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
                )

        # Ensure downstream consumers (PositionManager, logs) see the normalized correlation_id.
        fill = dict(fill)
        fill["correlation_id"] = corr_id

        # Persist fill row when we can map it to an Order row.
        if self.db:
            self._persist_fill(fill, corr_id, fill_ts, fill_price, fill_qty)

        # PRD 6.7: PositionManager is the source of truth for position state updates on fills.
        try:
            decision = self.position_manager.on_fill(
                state,
                self.market_state,
                fill,
                risk_cfg=(
                    self.config.get("risk")
                    if isinstance(self.config.get("risk"), dict)
                    else None
                ),
            )
        except Exception as e:
            self._inc_error("execution")
            from src.core.errors import ErrorCode

            self.logger.error(
                "PositionManager on_fill failed",
                error_code=ErrorCode.POSITION_UPDATE_FAILED.value,
                symbol=symbol,
                correlation_id=corr or None,
                error=str(e),
                exc_info=True,
            )
            return

        self._log_position_decision(decision, state, fill)
        self._update_realized_pnl(decision, fill_ts)

        if decision.closed_trade is not None:
            closed = decision.closed_trade
            try:
                self.closed_trades.append(closed)
            except Exception:
                pass

            if self.db:
                self._persist_closed_trade(closed)

            try:
                self.risk_manager.record_completed_trade(
                    closed.strategy, as_of=closed.exit_time
                )
            except Exception:
                pass

    def _persist_fill(
        self,
        fill: Dict[str, Any],
        corr_id: str,
        fill_ts: datetime,
        fill_price: float,
        fill_qty: float,
    ) -> None:
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
                            time_placed=fill_ts,
                            time_last_update=fill_ts,
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

            # Idempotency
            existing = (
                session.query(DbFill)
                .filter(DbFill.order_id == int(order_id))
                .filter(DbFill.fill_time == fill_ts)
                .filter(DbFill.fill_price == fill_price)
                .filter(DbFill.fill_qty == fill_qty)
                .first()
            )
            if existing is not None:
                return

            session.add(
                DbFill(
                    order_id=order_id,
                    fill_time=fill_ts,
                    fill_price=fill_price,
                    fill_qty=fill_qty,
                )
            )

        if self.db:
            self.db.write("fill", _write_fill)

    def _log_position_decision(
        self, decision: Any, state: SymbolState, fill: Dict[str, Any]
    ) -> None:
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
                    decision.closed_trade.strategy
                    if decision.closed_trade is not None
                    else fill.get("strategy")
                ),
            )

    def _update_realized_pnl(self, decision: Any, fill_ts: Any) -> None:
        if decision.realized_pnl_delta:
            try:
                self.risk_manager.update_pnl(
                    float(decision.realized_pnl_delta),
                    as_of=(fill_ts if isinstance(fill_ts, datetime) else None),
                )
            except Exception:
                pass
            try:
                self.market_state.daily_pnl = float(self.risk_manager.current_daily_pnl)
            except Exception:
                pass

    def _persist_closed_trade(self, closed: Any) -> None:
        def _write_trade(session: Session) -> None:
            trade = DbTrade(
                symbol=closed.symbol,
                strategy=closed.strategy,
                regime_at_entry=closed.regime_at_entry,
                regime_at_exit=closed.regime_at_exit,
                side=closed.side,
                qty=closed.qty,
                entry_time=closed.entry_time,
                exit_time=closed.exit_time,
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
            from src.core.errors import ErrorCode

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

    async def run_scan(self):
        """
        Triggers a scan and updates the active symbol list/strategies.
        """
        if not self.scanner:
            self.logger.warning("Scanner not initialized")
            return

        try:
            results = await self.scanner.scan(
                regime=self.market_state.regime, scan_time=self.market_state.time
            )
        except Exception as e:
            self.error_counts["scanner"] += 1
            from src.core.errors import ErrorCode

            self.logger.error(
                "Scanner scan failed; retaining existing watchlist",
                error_code=ErrorCode.SCANNER_SCAN_FAILED.value,
                regime=getattr(
                    self.market_state.regime, "value", str(self.market_state.regime)
                ),
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

        final_adds, final_removes = self._calculate_scan_churn(
            scan_results, existing_non_index, index_symbol
        )

        new_target_map = {res.symbol: res for res in scan_results}

        self._apply_scan_diff(
            final_adds, final_removes, new_target_map, scan_results, existing_non_index
        )

    def _normalize_scan_results(
        self, scan_result: ScanResult | List[WatchlistSymbol]
    ) -> List[Any]:
        scan_results: List[Any]
        if isinstance(scan_result, ScanResult):
            scan_results = list(scan_result.watchlist)
        else:
            scan_results = list(scan_result)

        index_symbol = str(self.config.get("index_symbol", "SPY") or "")

        # PRD 1.1: keep total WS subscriptions within Alpaca's practical ~30 ticker limit.
        has_index_in_scan = any(
            getattr(res, "symbol", None) == index_symbol for res in scan_results
        )
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
        to_add = [
            s
            for s in desired_symbols
            if s not in existing_non_index and s != index_symbol
        ]
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
        self._update_retained_strategies(
            scan_results, existing_non_index, new_target_map
        )

    def _process_scan_removals(self, final_removes: List[str]) -> None:
        for sym in final_removes:
            # If a symbol falls out of the scan list, we drop state unless there's an open position.
            if self.symbol_states[sym].position:
                self.logger.info(
                    "Retaining symbol with position outside scan", symbol=sym
                )
                continue

            # Best-effort: cancel any locally-tracked open orders for this symbol when removed.
            if self.db and self.order_executor:
                try:
                    # Cancel any broker-open orders for this symbol (covers cases where DB misses rows).
                    try:
                        self.order_executor.cancel_all_for_symbol(sym)
                    except Exception:
                        pass

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
                        broker_ids = [
                            str(r.broker_order_id) for r in rows if r.broker_order_id
                        ]

                    self.db.write("collect_pending_cancel", _collect)

                    if broker_ids:
                        self.logger.info(
                            "Clean-up canceling open orders for removed symbol",
                            symbol=sym,
                            count=len(broker_ids),
                        )
                        for oid in sorted(set(broker_ids)):
                            self.order_executor.cancel_by_broker_order_id(oid)
                except Exception as e:
                    self.logger.warning(
                        "Clean-up cancel failed", symbol=sym, error=str(e)
                    )

            self.logger.info("Dropping symbol from active set", symbol=sym)
            del self.symbol_states[sym]
            if self.alpaca_client:
                self.alpaca_client.unsubscribe(sym)

    def _process_scan_additions(
        self, final_adds: List[str], new_target_map: Dict[str, Any]
    ) -> None:
        for sym in final_adds:
            scanned_sym = new_target_map[sym]
            self.logger.info("Adding symbol to active set", symbol=sym)

            # Initialize SymbolState
            self.symbol_states[sym] = SymbolState(
                symbol=sym,
                bars=deque(maxlen=100),
                position=None,
                indicators={},
                open_orders={},
                allowed_strategies=list(scanned_sym.strategies),
                meta={},
            )

            # Pre-hydrate features from scan result
            flow_bias = "neutral"
            features = scanned_sym.features
            if features and features.extra:
                prem = float(features.extra.get("flow_total_premium") or 0.0)
                if prem > 1000000:
                    flow_bias = "bullish"
                elif prem < -1000000:
                    flow_bias = "bearish"

            features_snapshot = {}
            if features:
                features_snapshot = self._sanitize_features_snapshot(asdict(features))

            earnings_date = None
            sector = "unknown"
            iv_rank = 0.0
            short_interest = 0.0
            call_put_ratio = 0.0
            premarket_volume = 0.0
            flow_zscore = 0.0
            gap_pct = 0.0

            if features:
                sector = str(features.extra.get("sector") or "unknown")
                earnings_date = features.extra.get("earnings_date")
                iv_rank = float(features.extra.get("iv_rank") or 0.0)
                short_interest = float(features.extra.get("short_interest") or 0.0)
                call_put_ratio = features.call_put_ratio
                premarket_volume = features.premarket_volume
                flow_zscore = features.flow_zscore
                gap_pct = features.gap_pct
                if earnings_date:
                    earnings_date = str(earnings_date)

            self.symbol_states[sym].meta.update(
                {
                    "added_at": str(self.market_state.time),
                    "initial_score": float(scanned_sym.score),
                    "sector": sector,
                    "earnings_date": earnings_date,
                    "iv_rank": iv_rank,
                    "short_interest": short_interest,
                    "call_put_ratio": call_put_ratio,
                    "flow_bias": flow_bias,
                    "flow_zscore": flow_zscore,
                    "gap_pct": gap_pct,
                    "premarket_volume": premarket_volume,
                    "features_snapshot": features_snapshot,
                }
            )

            if self.alpaca_client:
                self.alpaca_client.subscribe(sym)

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
                self.symbol_states[sym].allowed_strategies = list(
                    scanned_sym.strategies
                )
                if scanned_sym.features:
                    features = scanned_sym.features
                    if features.extra:
                        self.symbol_states[sym].meta["flow_bias"] = float(
                            features.extra.get("flow_bias") or 0.0
                        )
                    self.symbol_states[sym].meta[
                        "premarket_volume"
                    ] = features.premarket_volume
                    self.symbol_states[sym].meta["features_snapshot"] = (
                        self._sanitize_features_snapshot(asdict(features))
                    )

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
        alpaca_client = self.alpaca_client
        if alpaca_client is None:
            return None

        trading_client = alpaca_client.trading_client
        import asyncio

        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        def _get_orders_list(
            status: QueryOrderStatus, limit: Optional[int] = None
        ) -> List[Any]:
            req = GetOrdersRequest(status=status, limit=limit)
            resp = trading_client.get_orders(req)
            return resp if isinstance(resp, list) else []

        try:
            account = await asyncio.to_thread(trading_client.get_account)  # type: ignore
            positions = await asyncio.to_thread(trading_client.get_all_positions)
            orders = await asyncio.to_thread(_get_orders_list, QueryOrderStatus.OPEN)
        except Exception as e:
            self.error_counts["orders"] += 1
            from src.core.errors import ErrorCode

            self.logger.error(
                "Reconcile fetch failed",
                error_code=ErrorCode.RECONCILE_FETCH_FAILED.value,
                error=str(e),
            )
            return None

        # Best-effort: also fetch recently closed orders
        closed_orders: List[Any] = []
        try:
            closed_orders = await asyncio.to_thread(
                _get_orders_list, QueryOrderStatus.CLOSED, 200
            )
        except Exception:
            closed_orders = []

        return account, positions, orders, closed_orders

    def _reconcile_positions(
        self, positions: Any, broker_position_symbols: set[str]
    ) -> None:
        if not positions:
            positions = []

        for p in positions:
            try:
                sym = str(getattr(p, "symbol", ""))
                if not sym:
                    continue
                broker_position_symbols.add(sym)
                side_raw = getattr(p, "side", "")
                side = Side.LONG if str(side_raw).lower() == "long" else Side.SHORT
                qty = float(getattr(p, "qty", getattr(p, "quantity", 0.0)) or 0.0)
                avg_price = float(
                    getattr(p, "avg_entry_price", getattr(p, "avg_price", 0.0)) or 0.0
                )

                state = self.symbol_states.get(sym)
                if state is None:
                    state = SymbolState(
                        symbol=sym,
                        bars=deque(maxlen=100),
                        position=None,
                        indicators={},
                        open_orders={},
                        allowed_strategies=[],
                        meta={},
                    )
                    self.symbol_states[sym] = state

                state.position = Position(
                    symbol=sym,
                    side=side,
                    qty=qty,
                    avg_price=avg_price,
                    unrealized_pnl=0.0,
                    realized_pnl=(
                        getattr(state.position, "realized_pnl", 0.0)
                        if state.position
                        else 0.0
                    ),
                    strategy=(state.position.strategy if state.position else "unknown"),
                    entry_time=getattr(state.position, "entry_time", None),
                    correlation_id=(
                        getattr(state.position, "correlation_id", "")
                        if state.position
                        else ""
                    ),
                    regime_at_entry=getattr(state.position, "regime_at_entry", None),
                )
            except Exception as e:
                self.logger.warning("Position reconcile failed", error=str(e))

        # Detect local positions missing at broker
        missing = []
        for sym, st in self.symbol_states.items():
            if st.position is None:
                continue
            if sym not in broker_position_symbols:
                missing.append(sym)

        if missing:
            mode = str(self.config.get("position_mismatch_mode", "halt")).lower()
            self.logger.error(
                "Position mismatch detected (local position missing at broker)",
                symbols=sorted(missing),
                mode=mode,
            )
            if mode in ("halt", "stop"):
                self._set_risk_mode("off")

    def _reconcile_orders(
        self, orders: Any, closed_orders: Any, broker_order_ids: set[str]
    ) -> None:
        if not orders:
            orders = []

        # Reconcile open orders into symbol state meta
        for o in orders:
            try:
                sym = str(getattr(o, "symbol", ""))
                oid = str(getattr(o, "id", ""))
                broker_order_ids.add(oid)
                if not sym or not oid:
                    continue
                state = self.symbol_states.get(sym)
                if state is None:
                    state = SymbolState(
                        symbol=sym,
                        bars=deque(maxlen=100),
                        position=None,
                        indicators={},
                        open_orders={},
                        allowed_strategies=[],
                        meta={},
                    )
                    self.symbol_states[sym] = state
                state.open_orders[oid] = {
                    "status": str(getattr(o, "status", "")),
                    "qty": float(getattr(o, "qty", getattr(o, "quantity", 0.0)) or 0.0),
                    "side": str(getattr(o, "side", "")),
                    "limit_price": getattr(o, "limit_price", None),
                    "created_at": str(getattr(o, "created_at", "") or ""),
                }
            except Exception as e:
                self.logger.warning("Order reconcile failed", error=str(e))

        # Deterministic stale-order cancel (optional).
        max_age_sec = int(self.config.get("max_open_order_age_sec", 0))
        if max_age_sec > 0 and self.order_executor:
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

        # DB reconciliation
        if self.db:
            self._reconcile_db_orders(orders, closed_orders, broker_order_ids)

        # Synthesize missed fills
        if self.db and closed_orders:
            self._synthesize_missed_fills(closed_orders)

    def _reconcile_db_orders(
        self, orders: Any, closed_orders: Any, broker_order_ids: set[str]
    ) -> None:
        now = self.clock()

        def _reconcile_db(session: Session) -> None:
            # Update statuses for orders still open at broker
            for o in orders or []:
                oid = str(getattr(o, "id", ""))
                status_val = str(getattr(o, "status", "") or "")
                if not oid:
                    continue

                row = (
                    session.query(DbOrder)
                    .filter(DbOrder.broker_order_id == oid)
                    .order_by(DbOrder.id.desc())
                    .first()
                )
                if row:
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

            # Update statuses for recently closed orders (helps correct broker_missing mislabels).
            for o in closed_orders or []:
                oid = str(getattr(o, "id", "") or "")
                status_val = str(getattr(o, "status", "") or "")
                if not oid or not status_val:
                    continue
                row = (
                    session.query(DbOrder)
                    .filter(DbOrder.broker_order_id == oid)
                    .order_by(DbOrder.id.desc())
                    .first()
                )
                if row:
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

            # Mark DB orders as cancelled if broker no longer has them open.
            # Using broker_order_ids to check set membership
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

        if self.db:
            try:
                self.db.write("order_reconcile", _reconcile_db)
            except Exception as e:
                self.logger.warning("DB reconcile failed", error=str(e))

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
            order_id = self._get_reconciled_order_id(
                broker_id, corr, sym, side_val, filled_qty, o, fill_ts
            )

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
            self._persist_synthesized_fill(
                order_id, fill_ts, filled_avg_price, filled_qty
            )

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
                session.query(DbOrder)
                .filter(DbOrder.broker_order_id == broker_id)
                .order_by(DbOrder.id.desc())
                .first()
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
            existing = (
                session.query(DbFill)
                .filter(DbFill.order_id == int(row.id))
                .order_by(DbFill.id.desc())
                .first()
            )
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
            await self.reconcile_broker_state()
