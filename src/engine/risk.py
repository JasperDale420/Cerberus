from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.core.domain import MarketState, OrderIntent, OrderType, Signal, SymbolState
from src.core.logger import StructuredLogger


class RiskManager:
    """
    Enforces risk limits and converts Signals to OrderIntents.
    """

    DEFAULT_TIMEZONE = "US/Eastern"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        self.config = config
        self.logger = logger
        risk_cfg = (
            config.get("risk") if isinstance(config.get("risk"), dict) else None
        ) or {}

        # Account-level / global risk (prefer nested risk.yaml config; fallback to flat keys)
        self.max_daily_loss = float(
            risk_cfg.get("max_daily_loss", config.get("max_daily_loss", 1000.0))
        )
        self.max_risk_per_trade = float(
            risk_cfg.get("max_risk_per_trade", config.get("max_risk_per_trade", 50.0))
        )  # dollars risk = |entry-stop| * qty
        self.max_open_risk = float(
            risk_cfg.get("max_open_risk", config.get("max_open_risk", 0.0))
        )
        self.max_trades_per_day = int(
            risk_cfg.get("max_trades_per_day", config.get("max_trades_per_day", 0))
        )
        self.max_trades_per_strategy = int(
            risk_cfg.get(
                "max_trades_per_strategy", config.get("max_trades_per_strategy", 0)
            )
        )
        self.risk_mode = str(
            risk_cfg.get("risk_mode", config.get("risk_mode", "normal"))
        ).lower()

        # Non-PRD limits retained for safety/backward compatibility
        self.max_orders_per_day = int(config.get("max_orders_per_day", 100))
        self.max_open_positions = int(
            risk_cfg.get("max_open_positions", config.get("max_open_positions", 5))
        )
        self.max_notional_per_order = float(
            risk_cfg.get(
                "max_notional_per_order", config.get("max_notional_per_order", 5000.0)
            )
        )
        self.max_notional_per_symbol = float(
            risk_cfg.get(
                "max_notional_per_symbol", config.get("max_notional_per_symbol", 0.0)
            )
        )

        self.current_daily_pnl = 0.0
        self.daily_order_count = 0
        # PRD 6.5: "max_trades_per_day/per strategy" is enforced on accepted entry attempts
        # (i.e., approved signals), not completed round-trip trades.
        self.daily_entry_count = 0
        self.per_strategy_entry_count: Dict[str, int] = {}

        # Completed trades are tracked separately for reporting/observability.
        self.daily_completed_trade_count = 0
        self.per_strategy_completed_trade_count: Dict[str, int] = {}

        # Deterministic daily rollover keyed to market/session date.
        self._session_date: Optional[date] = None
        self.last_rejection_reason: Optional[str] = None

    def _session_date_for(self, as_of: datetime) -> date:
        # Use configured timezone when available; default to US equities session time.
        tz_name = str(
            self.config.get("timezone", self.DEFAULT_TIMEZONE) or self.DEFAULT_TIMEZONE
        )
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo(self.DEFAULT_TIMEZONE)

        dt = as_of
        if isinstance(dt, datetime) and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(tz).date()

    def _maybe_rollover(self, as_of: Optional[datetime]) -> None:
        """
        Deterministic daily reset based on market time (PRD 11.1).

        This allows multi-day backtests/replays without requiring a process restart.
        """
        if not isinstance(as_of, datetime):
            return
        session_date = self._session_date_for(as_of)
        if self._session_date is None:
            self._session_date = session_date
            return
        if session_date == self._session_date:
            return

        # New session/day: reset daily counters deterministically.
        self._session_date = session_date
        self.current_daily_pnl = 0.0
        self.daily_order_count = 0
        self.daily_entry_count = 0
        self.daily_completed_trade_count = 0
        self.per_strategy_entry_count.clear()
        self.per_strategy_completed_trade_count.clear()
        self.last_rejection_reason = None

    def _check_strategy_config(self, signal: Signal) -> Optional[Dict[str, Any]]:
        strat_cfg = None
        if isinstance(self.config.get("strategies"), dict):
            strat_cfg = (self.config.get("strategies") or {}).get(signal.strategy)

        if isinstance(strat_cfg, dict):
            enabled = bool(strat_cfg.get("enabled", True))
            regimes_cfg = strat_cfg.get("regimes")
            if isinstance(regimes_cfg, dict):
                r_key = getattr(signal.regime, "value", str(signal.regime))
                r_cfg = regimes_cfg.get(r_key)
                if isinstance(r_cfg, dict) and "enabled" in r_cfg:
                    enabled = bool(r_cfg.get("enabled"))

            if not enabled:
                return None
        return strat_cfg if isinstance(strat_cfg, dict) else {}

    def _check_basic_gates(self, strat_cfg: Optional[Dict[str, Any]]) -> bool:
        if strat_cfg is None:
            self.last_rejection_reason = "STRATEGY_DISABLED"
            return False

        if self.risk_mode == "off":
            self.last_rejection_reason = "RISK_MODE_OFF"
            return False

        return True

    def _check_volume_gates(self, signal: Signal) -> bool:
        if (
            self.max_trades_per_day > 0
            and self.daily_entry_count >= self.max_trades_per_day
        ):
            self.last_rejection_reason = "MAX_TRADES_PER_DAY"
            return False

        if self.max_trades_per_strategy > 0:
            strat_entries = self.per_strategy_entry_count.get(signal.strategy, 0)
            if strat_entries >= self.max_trades_per_strategy:
                self.last_rejection_reason = "MAX_STRAT_TRADES"
                return False

        if self.daily_order_count >= self.max_orders_per_day:
            self.last_rejection_reason = "MAX_DAILY_ORDERS"
            return False

        return True

    def _check_position_gates(
        self,
        signal: Signal,
        symbol_state: SymbolState,
        current_positions: Optional[List[Any]],
    ) -> bool:
        if current_positions is None or symbol_state.position is not None:
            return True

        if len(current_positions) >= self.max_open_positions:
            self.last_rejection_reason = "MAX_POSITIONS"
            return False

        risk_cfg = (
            self.config.get("risk")
            if isinstance(self.config.get("risk"), dict)
            else None
        ) or {}
        max_strat_pos = int(
            risk_cfg.get(
                "max_positions_per_strategy",
                self.config.get("max_positions_per_strategy", 3),
            )
        )
        strat_positions = [
            p
            for p in current_positions
            if getattr(p, "strategy", "") == signal.strategy
        ]

        if len(strat_positions) >= max_strat_pos:
            self.last_rejection_reason = "MAX_STRAT_POSITIONS"
            return False

        return True

    def _check_loss_gate(self) -> bool:
        if self.current_daily_pnl <= -self.max_daily_loss:
            self.last_rejection_reason = "MAX_DAILY_LOSS"
            return False
        return True

    def _apply_risk_mode(self, effective_max_risk: float) -> float:
        if self.risk_mode == "reduced":
            return effective_max_risk * 0.5
        return effective_max_risk

    def _apply_strategy_limits(
        self, effective_max_risk: float, signal: Signal, strat_cfg: Dict[str, Any]
    ) -> float:
        if strat_cfg.get("max_risk_per_trade") is not None:
            try:
                effective_max_risk = min(
                    effective_max_risk, float(strat_cfg["max_risk_per_trade"])
                )
            except Exception:
                pass

        regimes_cfg = strat_cfg.get("regimes")
        if isinstance(regimes_cfg, dict):
            r_key = getattr(signal.regime, "value", str(signal.regime))
            r_cfg = regimes_cfg.get(r_key)
            if isinstance(r_cfg, dict) and r_cfg.get("max_risk_per_trade") is not None:
                try:
                    effective_max_risk = min(
                        effective_max_risk, float(r_cfg["max_risk_per_trade"])
                    )
                except Exception:
                    pass
        return effective_max_risk

    def _apply_equity_constraints(
        self,
        qty_limit: int,
        signal: Signal,
        account_equity: float,
        risk_per_share: float,
        effective_max_risk: float,
    ) -> int:
        if account_equity > 0 and signal.entry_price > 0:
            max_equity_allocation = account_equity * 0.05
            max_qty_equity = int(max_equity_allocation / signal.entry_price)
            if max_qty_equity < qty_limit:
                self.logger.info(
                    "Risk sizing constrained by equity",
                    symbol=signal.symbol,
                    equity=account_equity,
                    max_equity_qty=max_qty_equity,
                    risk_qty=int(effective_max_risk / risk_per_share),
                )
                return max_qty_equity
        return qty_limit

    def _calculate_qty(
        self, signal: Signal, account_equity: float, strat_cfg: Dict[str, Any]
    ) -> int:
        risk_per_share = abs(signal.entry_price - signal.stop_price)
        if risk_per_share <= 0:
            self.last_rejection_reason = "INVALID_STOP"
            return 0

        effective_max_risk = self.max_risk_per_trade
        effective_max_risk = self._apply_risk_mode(effective_max_risk)
        effective_max_risk = self._apply_strategy_limits(
            effective_max_risk, signal, strat_cfg
        )

        qty_limit = int(effective_max_risk / risk_per_share)
        qty_limit = self._apply_equity_constraints(
            qty_limit, signal, account_equity, risk_per_share, effective_max_risk
        )

        qty: int = 0
        if signal.size_hint:
            qty = min(int(signal.size_hint), qty_limit)
        else:
            qty = qty_limit

        if qty <= 0:
            self.last_rejection_reason = "ZERO_QTY"
            return 0

        return qty

    def _check_notional_limits(
        self, notional: float, symbol_state: SymbolState
    ) -> bool:
        if notional > self.max_notional_per_order:
            self.last_rejection_reason = "MAX_NOTIONAL"
            return False

        if self.max_notional_per_symbol > 0:
            existing_notional = 0.0
            if symbol_state.position:
                existing_notional = float(
                    symbol_state.position.qty * symbol_state.position.avg_price
                )
            if (existing_notional + notional) > self.max_notional_per_symbol:
                self.last_rejection_reason = "MAX_SYMBOL_NOTIONAL"
                return False
        return True

    def _check_risk_exposure(
        self, risk_per_share: float, qty: int, current_positions: Optional[List[Any]]
    ) -> bool:
        if self.max_open_risk > 0 and current_positions is not None:
            open_risk = 0.0
            for p in current_positions:
                open_risk += float(getattr(p, "open_risk", 0.0) or 0.0)
            proposed_risk = risk_per_share * qty
            if (open_risk + proposed_risk) > self.max_open_risk:
                self.last_rejection_reason = "MAX_OPEN_RISK"
                return False
        return True

    def _create_intent(self, signal: Signal, qty: int) -> OrderIntent:
        risk_cfg = (
            self.config.get("risk")
            if isinstance(self.config.get("risk"), dict)
            else None
        ) or {}
        time_in_force = str(risk_cfg.get("time_in_force", "day") or "day")
        return OrderIntent(
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            order_type=OrderType.LIMIT,
            limit_price=signal.entry_price,
            time_in_force=time_in_force,
            correlation_id=signal.correlation_id,
            stop_loss=signal.stop_price,
            take_profit=signal.target_price,
            strategy=signal.strategy,
            meta={"created_at": signal.generated_at.isoformat()},
        )

    def apply(
        self,
        signal: Signal,
        symbol_state: SymbolState,
        market_state: MarketState,
        current_positions: Optional[List[Any]] = None,
        account_equity: float = 0.0,
    ) -> List[OrderIntent]:
        """
        Evaluates a signal and returns a list of OrderIntents if approved, or an empty list if rejected.
        """
        self._maybe_rollover(getattr(market_state, "time", None))
        self.last_rejection_reason = None

        strat_cfg = self._check_strategy_config(signal)
        # Removed unused 'signal' parameter
        if not self._check_basic_gates(strat_cfg):
            self._log_rejection(signal)
            return []

        if not self._check_volume_gates(signal):
            self._log_rejection(signal)
            return []

        if not self._check_position_gates(signal, symbol_state, current_positions):
            self._log_rejection(signal)
            return []

        if not self._check_loss_gate():
            self._log_rejection(signal)
            return []

        # Calculate Qty
        qty = self._calculate_qty(signal, account_equity, strat_cfg or {})
        if qty <= 0:
            self._log_rejection(signal)
            return []

        notional = qty * signal.entry_price
        if not self._check_notional_limits(notional, symbol_state):
            self._log_rejection(signal)
            return []

        risk_per_share = abs(signal.entry_price - signal.stop_price)
        if not self._check_risk_exposure(risk_per_share, qty, current_positions):
            self._log_rejection(signal)
            return []

        intent = self._create_intent(signal, qty)

        self.daily_order_count += 1
        self.daily_entry_count += 1
        self.per_strategy_entry_count[signal.strategy] = (
            self.per_strategy_entry_count.get(signal.strategy, 0) + 1
        )
        self.logger.info(
            "Signal approved",
            symbol=intent.symbol,
            strategy=intent.strategy,
            correlation_id=intent.correlation_id,
            regime=getattr(signal.regime, "value", str(signal.regime)),
            intent=intent,
            daily_order_count=self.daily_order_count,
            daily_entry_count=self.daily_entry_count,
        )
        return [intent]

    def _log_rejection(self, signal: Signal) -> None:
        self.logger.warning(
            f"Signal rejected: {self.last_rejection_reason}",
            symbol=signal.symbol,
            strategy=signal.strategy,
            correlation_id=getattr(signal, "correlation_id", "") or None,
            regime=getattr(signal.regime, "value", str(signal.regime)),
            reason_code=self.last_rejection_reason,
        )

    def update_pnl(self, pnl: float, *, as_of: Optional[datetime] = None) -> None:
        """
        Updates the current daily PnL.
        """
        self._maybe_rollover(as_of)
        self.current_daily_pnl += pnl

    def record_completed_trade(
        self, strategy: str, *, as_of: Optional[datetime] = None
    ) -> None:
        """Increment completed-trade counters (not used for entry caps)."""
        self._maybe_rollover(as_of)
        self.daily_completed_trade_count += 1
        self.per_strategy_completed_trade_count[strategy] = (
            self.per_strategy_completed_trade_count.get(strategy, 0) + 1
        )
