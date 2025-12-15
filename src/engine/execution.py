from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.analysis.db import DatabaseDatabase
from src.analysis.regime import Regime, RegimeDetector
from src.analysis.schema import Signal as DbSignal
from src.analysis.schema import Trade as DbTrade
from src.core.domain import Position, Side
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.engine.orders import OrderExecutor
from src.engine.risk import RiskManager
from src.scanner.core import Scanner
from src.strategies.base import BaseStrategy, MarketState, Signal, SymbolState


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
    ):
        self.config = config
        self.logger = logger
        self.db = db
        self.alpaca_client = alpaca_client
        self.run_id = run_id

        self.regime_detector = RegimeDetector(logger=logger)
        self.risk_manager = RiskManager(config, logger)
        self.order_executor = (
            OrderExecutor(alpaca_client, logger, db) if alpaca_client else None
        )
        self.scanner: Optional[Scanner] = None  # Injected later or via init

        self.strategies: Dict[str, BaseStrategy] = {}
        self.symbol_states: Dict[str, SymbolState] = {}
        self.market_state = MarketState(
            time=datetime.now(timezone.utc), regime=Regime.CHOP
        )

        # Throttling config
        self.max_churn_per_scan = config.get("max_churn_per_scan", 2)

    def register_strategy(self, strategy: BaseStrategy):
        """
        Registers a strategy with the engine.
        """
        self.strategies[strategy.name] = strategy
        self.logger.info("Strategy registered", strategy=strategy.name)

    def on_bar(self, symbol: str, bar: Any):
        """
        Process a new bar for a symbol.
        """
        # Update Market State (if symbol is index)
        if symbol == self.config.get("index_symbol", "SPY"):
            self.market_state.regime = self.regime_detector.update(bar.close)
            # FIX: Use bar time for determinism
            self.market_state.time = getattr(
                bar, "time", getattr(bar, "timestamp", datetime.now(timezone.utc))
            )

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

        # Run Strategies
        for name, strategy in self.strategies.items():
            # Check if strategy is allowed for this symbol
            if name in state.allowed_strategies:
                try:
                    signal = strategy.on_bar(symbol, bar, state, self.market_state)
                    if signal:
                        self.logger.info("Signal generated", signal=signal)
                        self._process_signal(signal)
                except Exception as e:
                    self.logger.error(
                        "Strategy execution failed",
                        strategy=name,
                        symbol=symbol,
                        error=str(e),
                    )

    def _process_signal(self, signal: Signal):
        """
        Passes signal to Risk Manager and then Order Executor.
        Persists signal to DB.
        """
        # Inject run_id if available and not present
        if self.run_id and not hasattr(signal, "run_id"):
            # Signal is a Pydantic model or dataclass, might not handle dynamic attrs easily if frozen
            # But let's assume we can rely on context logging
            pass

        self.logger.info("Processing signal", signal=signal, run_id=self.run_id)

        # 1. Risk Check
        import time  # Import time here to avoid circular dependency or global import if not needed elsewhere

        # Gather active positions from symbol states
        active_positions = [
            s.position for s in self.symbol_states.values() if s.position is not None
        ]

        start_risk = time.perf_counter()
        intent = self.risk_manager.apply(
            signal,
            self.symbol_states[signal.symbol],
            self.market_state,
            current_positions=active_positions,
        )
        risk_latency = (time.perf_counter() - start_risk) * 1000

        # 2. Persist Signal
        if self.db:
            try:
                with self.db.get_session() as session:
                    db_signal = DbSignal(
                        correlation_id=signal.correlation_id,
                        symbol=signal.symbol,
                        strategy=signal.strategy,
                        regime=signal.regime.value,
                        time=signal.generated_at,
                        raw_side=signal.side.value,
                        raw_size=signal.size_hint,
                        accepted=intent is not None,
                        rejection_reason=None if intent else "Risk Rejection",
                        meta_json=signal.meta,
                    )
                    session.add(db_signal)
            except Exception as e:
                self.logger.error("Failed to persist signal", error=str(e))

        if not intent:
            self.logger.info("Signal rejected by Risk Manager")
            return

        # 3. Order Execution
        if self.order_executor:
            try:
                start_exec = time.perf_counter()
                self.order_executor.submit(intent)
                exec_latency = (time.perf_counter() - start_exec) * 1000
                self.logger.info(
                    "Order submitted",
                    intent=intent,
                    risk_latency_ms=risk_latency,
                    exec_latency_ms=exec_latency,
                    run_id=self.run_id,
                )
            except Exception as e:
                self.logger.error(
                    "Order execution failed", error=str(e), run_id=self.run_id
                )
        else:
            self.logger.warning(
                "No OrderExecutor available (paper/mock mode)",
                intent=intent,
                run_id=self.run_id,
            )

    def on_fill(self, fill: Dict[str, Any]):
        """
        Updates position state and persists Trades on close.
        fill struct: {
            symbol, side (buy/sell), qty, price,
            order_id, timestamp, strategy, correlation_id
        }
        """
        symbol = fill["symbol"]
        if symbol not in self.symbol_states:
            self.logger.warning("Fill received for unknown symbol", symbol=symbol)
            return

        state = self.symbol_states[symbol]
        fill_qty = float(fill["qty"])
        fill_price = float(fill["price"])
        fill_side = fill["side"]  # "buy" or "sell"

        # Determine direction
        # Update or Create Position
        if state.position is None:
            # New Position
            side = Side.LONG if fill_side == "buy" else Side.SHORT
            state.position = Position(
                symbol=symbol,
                side=side,
                qty=fill_qty,
                avg_price=fill_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                strategy=fill.get("strategy", "unknown"),
            )
            self.logger.info("Position opened", position=state.position)

        else:
            # Update Existing
            pos = state.position

            # Check if increasing or reducing
            is_same_side = (pos.side == Side.LONG and fill_side == "buy") or (
                pos.side == Side.SHORT and fill_side == "sell"
            )

            if is_same_side:
                # Increasing
                total_cost = (pos.qty * pos.avg_price) + (fill_qty * fill_price)
                total_qty = pos.qty + fill_qty
                pos.avg_price = total_cost / total_qty
                pos.qty = total_qty
                self.logger.info("Position increased", position=pos)
            else:
                # Reducing / Closing
                # Calculate realized PnL on the closed portion
                close_qty = min(pos.qty, fill_qty)

                if pos.side == Side.LONG:
                    pnl = (fill_price - pos.avg_price) * close_qty
                else:
                    pnl = (pos.avg_price - fill_price) * close_qty

                pos.realized_pnl += pnl
                self.risk_manager.update_pnl(pnl)  # Update daily limit

                pos.qty -= close_qty

                # If closed completely
                if pos.qty <= 0:
                    self.logger.info("Position closed", pnl=pnl, strategy=pos.strategy)

                    # Persist Closed Trade
                    if self.db:
                        try:
                            with self.db.get_session() as session:
                                db_trade = DbTrade(
                                    symbol=symbol,
                                    strategy=pos.strategy,
                                    regime_at_entry=fill.get(
                                        "regime", "unknown"
                                    ),  # Ideally tracked in Position
                                    regime_at_exit=self.market_state.regime.value,
                                    side=pos.side.value,
                                    qty=fill_qty,  # Total qty logic needed for accurate trade matching, simplified here
                                    entry_time=fill[
                                        "timestamp"
                                    ],  # Placeholder, should be pos start time
                                    exit_time=datetime.now(timezone.utc),
                                    entry_price=pos.avg_price,  # Initial avg
                                    exit_price=fill_price,
                                    pnl_gross=pnl,
                                    pnl_net=pnl,  # commission logic omitted
                                    correlation_id=fill.get("correlation_id", ""),
                                )
                                session.add(db_trade)
                        except Exception as e:
                            self.logger.error("Failed to persist trade", error=str(e))

                    state.position = None
                else:
                    self.logger.info("Position reduced", position=pos, pnl=pnl)

    async def run_scan(self):
        """
        Triggers a scan and updates the active symbol list/strategies.
        """
        if not self.scanner:
            self.logger.warning("Scanner not initialized")
            return

        # Ideally Scanner.scan returns ScanResult, but current implementation returns List[ScannedSymbol]
        # I need to adapt based on current scanner.core.py or update it.
        # domain.py defines ScanResult separately.
        # For now, I'll assume current scanner returns List of objects with .symbol, .matching_strategies

        results = await self.scanner.scan()

        # Convert to ScanResult structure expected by apply_scan_result for logic consistency
        # Assuming results is List[ScannedSymbol]

        # results is a ScanResult object
        self.apply_scan_result(results.watchlist)

    def apply_scan_result(self, scan_results: List[Any]):
        """
        Applies scan results with churn throttling.
        scan_results: List of ScannedSymbol (from scanner/core.py)
        """
        current_symbols = set(self.symbol_states.keys())
        # Always keep index symbol
        index_symbol = self.config.get("index_symbol", "SPY")

        new_target_map = {res.symbol: res for res in scan_results}
        new_symbols_set = set(new_target_map.keys())

        if index_symbol in current_symbols:
            current_symbols.remove(index_symbol)

        # 1. Identify Candidates for Add/Remove
        to_add = new_symbols_set - current_symbols
        to_remove = current_symbols - new_symbols_set

        # 2. Throttle Additions
        adds_allowed = min(len(to_add), self.max_churn_per_scan)
        final_adds = list(to_add)[:adds_allowed]

        # 3. Throttle Removals
        # (For now we can be symmetrical or just allow removals to match adds to keep size stable)
        removes_allowed = min(len(to_remove), self.max_churn_per_scan)
        final_removes = list(to_remove)[:removes_allowed]

        self.logger.info(
            "Applying Scan Churn",
            requested_add=len(to_add),
            actual_add=len(final_adds),
            requested_remove=len(to_remove),
            actual_remove=len(final_removes),
        )

        # 4. Apply Changes

        # Removes
        for sym in final_removes:
            # We might want to close positions here or keep tracking them until close.
            # PRD implies we unsubscribe.
            # If we have open position, we should logically keep it in a "closing only" state?
            # For this slice, we just delete state if no position, or warn.
            if self.symbol_states[sym].position:
                self.logger.info(
                    "Keeping symbol with position despite scan remove", symbol=sym
                )
                continue

            del self.symbol_states[sym]
            if self.alpaca_client:
                self.alpaca_client.unsubscribe(sym)
                pass

        # Adds
        for sym in final_adds:
            scanned_sym = new_target_map[sym]
            self.symbol_states[sym] = SymbolState(
                symbol=sym,
                bars=deque(maxlen=100),
                position=None,
                indicators={},
                open_orders={},
                allowed_strategies=scanned_sym.strategies,
                meta={
                    "score": scanned_sym.score,
                    "gap_pct": (
                        scanned_sym.features.gap_pct if scanned_sym.features else 0.0
                    ),
                },
            )

            if self.alpaca_client:
                self.alpaca_client.subscribe(sym)
                pass

        # 5. Update strategies for retained symbols
        for sym in current_symbols & new_symbols_set:
            scanned_sym = new_target_map[sym]
            self.symbol_states[sym].allowed_strategies = scanned_sym.strategies
            # Also update meta
            if scanned_sym.features:
                self.symbol_states[sym].meta["gap_pct"] = scanned_sym.features.gap_pct
