from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.core.domain import MarketState, OrderIntent, OrderType, Signal, SymbolState
from src.core.logger import StructuredLogger


class RiskManager:
    """
    Enforces risk limits and converts Signals to OrderIntents.
    """

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
        tz_name = str(self.config.get("timezone", "US/Eastern") or "US/Eastern")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("US/Eastern")

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

    def apply(
        self,
        signal: Signal,
        symbol_state: SymbolState,
        market_state: MarketState,
        current_positions: Optional[List[Any]] = None,
    ) -> List[OrderIntent]:
        """
        Evaluates a signal and returns a list of OrderIntents if approved, or an empty list if rejected.

        PRD 6.5: `RiskManager.apply(...) -> List[OrderIntent]` (entry + optional OCO/exit intents).
        This implementation returns a single entry intent; broker-managed OCO exits are encoded via
        `OrderIntent.stop_loss` and `OrderIntent.take_profit` and submitted as a bracket order by
        `src/engine/orders.py`.
        """
        self._maybe_rollover(getattr(market_state, "time", None))
        self.last_rejection_reason = None

        # 0. Strategy enable gate (supports per-regime overrides in strategies.auto.yaml)
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
                self.last_rejection_reason = "STRATEGY_DISABLED"
                self.logger.warning(
                    "Signal rejected: Strategy disabled",
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    correlation_id=getattr(signal, "correlation_id", "") or None,
                    regime=getattr(signal.regime, "value", str(signal.regime)),
                    reason_code=self.last_rejection_reason,
                )
                return []

        # 0. Risk mode gate (entries only in this system)
        if self.risk_mode == "off":
            self.last_rejection_reason = "RISK_MODE_OFF"
            self.logger.warning(
                "Signal rejected: Risk mode OFF",
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=getattr(signal, "correlation_id", "") or None,
                regime=getattr(signal.regime, "value", str(signal.regime)),
                reason_code=self.last_rejection_reason,
            )
            return []

        # 0a. Max trades per day (completed trades, not orders)
        if (
            self.max_trades_per_day > 0
            and self.daily_entry_count >= self.max_trades_per_day
        ):
            self.last_rejection_reason = "MAX_TRADES_PER_DAY"
            self.logger.warning(
                "Signal rejected: Max trades per day exceeded",
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=getattr(signal, "correlation_id", "") or None,
                regime=getattr(signal.regime, "value", str(signal.regime)),
                count=self.daily_entry_count,
                limit=self.max_trades_per_day,
                reason_code=self.last_rejection_reason,
            )
            return []

        if self.max_trades_per_strategy > 0:
            strat_entries = self.per_strategy_entry_count.get(signal.strategy, 0)
            if strat_entries >= self.max_trades_per_strategy:
                self.last_rejection_reason = "MAX_STRAT_TRADES"
                self.logger.warning(
                    "Signal rejected: Max trades per strategy exceeded",
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    correlation_id=getattr(signal, "correlation_id", "") or None,
                    regime=getattr(signal.regime, "value", str(signal.regime)),
                    count=strat_entries,
                    limit=self.max_trades_per_strategy,
                    reason_code=self.last_rejection_reason,
                )
                return []

        # 0. Check Order Count Limit
        if self.daily_order_count >= self.max_orders_per_day:
            self.last_rejection_reason = "MAX_DAILY_ORDERS"
            self.logger.warning(
                "Signal rejected: Max daily orders exceeded",
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=getattr(signal, "correlation_id", "") or None,
                regime=getattr(signal.regime, "value", str(signal.regime)),
                count=self.daily_order_count,
                limit=self.max_orders_per_day,
                reason_code=self.last_rejection_reason,
            )
            return []

        # 0b. Check Global Position Limit
        # Only relevant for NEW entries (not exits or reductions)
        # Assuming signal.side matches entry direction (e.g. BUY for LONG entry)
        # We need to know if this creates a NEW position or adds to existing.
        # If symbol_state.position is None, it's a new position.
        if current_positions is not None and symbol_state.position is None:
            if len(current_positions) >= self.max_open_positions:
                self.last_rejection_reason = "MAX_POSITIONS"
                self.logger.warning(
                    "Signal rejected: Max open positions reached",
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    correlation_id=getattr(signal, "correlation_id", "") or None,
                    regime=getattr(signal.regime, "value", str(signal.regime)),
                    current=len(current_positions),
                    limit=self.max_open_positions,
                    reason_code=self.last_rejection_reason,
                )
                return []

            # 0c. Check Strategy Position Limit
            risk_cfg3 = (
                self.config.get("risk")
                if isinstance(self.config.get("risk"), dict)
                else None
            ) or {}
            max_strat_pos = int(
                risk_cfg3.get(
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
                self.logger.warning(
                    "Signal rejected: Max positions for strategy reached",
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    correlation_id=getattr(signal, "correlation_id", "") or None,
                    regime=getattr(signal.regime, "value", str(signal.regime)),
                    current=len(strat_positions),
                    limit=max_strat_pos,
                    reason_code=self.last_rejection_reason,
                )
                return []

        # 1. Check Daily Loss Limit
        if self.current_daily_pnl <= -self.max_daily_loss:
            self.last_rejection_reason = "MAX_DAILY_LOSS"
            self.logger.warning(
                "Signal rejected: Max daily loss exceeded",
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=getattr(signal, "correlation_id", "") or None,
                regime=getattr(signal.regime, "value", str(signal.regime)),
                current_pnl=self.current_daily_pnl,
                limit=self.max_daily_loss,
                reason_code=self.last_rejection_reason,
            )
            return []

        # Note: total open positions are enforced when `current_positions` is provided by the engine.

        # 2. Calculate Position Size based on Risk
        # Risk = |Entry - Stop| * Qty
        # Qty = MaxRisk / |Entry - Stop|

        risk_per_share = abs(signal.entry_price - signal.stop_price)
        if risk_per_share <= 0:
            self.last_rejection_reason = "INVALID_STOP"
            self.logger.warning(
                "Signal rejected: Invalid stop price (zero risk per share)",
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=getattr(signal, "correlation_id", "") or None,
                regime=getattr(signal.regime, "value", str(signal.regime)),
                reason_code=self.last_rejection_reason,
            )
            return []

        effective_max_risk = self.max_risk_per_trade
        if self.risk_mode == "reduced":
            effective_max_risk = self.max_risk_per_trade * 0.5

        # Per-strategy cap (PRD Stage 1 / config overrides)
        if (
            isinstance(strat_cfg, dict)
            and strat_cfg.get("max_risk_per_trade") is not None
        ):
            try:
                effective_max_risk = min(
                    effective_max_risk, float(strat_cfg["max_risk_per_trade"])
                )
            except Exception:
                pass
        if isinstance(strat_cfg, dict):
            regimes_cfg = strat_cfg.get("regimes")
            if isinstance(regimes_cfg, dict):
                r_key = getattr(signal.regime, "value", str(signal.regime))
                r_cfg = regimes_cfg.get(r_key)
                if (
                    isinstance(r_cfg, dict)
                    and r_cfg.get("max_risk_per_trade") is not None
                ):
                    try:
                        effective_max_risk = min(
                            effective_max_risk, float(r_cfg["max_risk_per_trade"])
                        )
                    except Exception:
                        pass

        qty_limit = effective_max_risk / risk_per_share
        qty_limit = int(qty_limit)  # Floor to be safe

        # If signal provides a size hint, respect it UP TO the limit
        if signal.size_hint:
            qty = min(int(signal.size_hint), qty_limit)
        else:
            qty = qty_limit

        if qty <= 0:
            self.last_rejection_reason = "ZERO_QTY"
            self.logger.warning(
                "Signal rejected: Calculated quantity is zero (Risk Limit exceeded)",
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=getattr(signal, "correlation_id", "") or None,
                regime=getattr(signal.regime, "value", str(signal.regime)),
                qty_limit=qty_limit,
                risk_per_share=risk_per_share,
                max_risk=self.max_risk_per_trade,
                reason_code=self.last_rejection_reason,
            )
            return []

        # 3. Check Notional Value
        notional = qty * signal.entry_price
        if notional > self.max_notional_per_order:
            self.last_rejection_reason = "MAX_NOTIONAL"
            self.logger.warning(
                "Signal rejected: Max notional per order exceeded",
                symbol=signal.symbol,
                strategy=signal.strategy,
                correlation_id=getattr(signal, "correlation_id", "") or None,
                regime=getattr(signal.regime, "value", str(signal.regime)),
                notional=notional,
                limit=self.max_notional_per_order,
                reason_code=self.last_rejection_reason,
            )
            return []

        # 3b. Symbol-level exposure (PRD "Max exposure per symbol") - defined as notional cap.
        if self.max_notional_per_symbol > 0:
            existing_notional = 0.0
            if symbol_state.position:
                existing_notional = float(
                    symbol_state.position.qty * symbol_state.position.avg_price
                )
            if (existing_notional + notional) > self.max_notional_per_symbol:
                self.last_rejection_reason = "MAX_SYMBOL_NOTIONAL"
                self.logger.warning(
                    "Signal rejected: Max notional per symbol exceeded",
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    correlation_id=getattr(signal, "correlation_id", "") or None,
                    regime=getattr(signal.regime, "value", str(signal.regime)),
                    existing_notional=existing_notional,
                    proposed_notional=notional,
                    limit=self.max_notional_per_symbol,
                    reason_code=self.last_rejection_reason,
                )
                return []

        # 3c. Account-level open risk cap (best-effort: sum position.open_risk if present).
        if self.max_open_risk > 0 and current_positions is not None:
            open_risk = 0.0
            for p in current_positions:
                open_risk += float(getattr(p, "open_risk", 0.0) or 0.0)
            proposed_risk = risk_per_share * qty
            if (open_risk + proposed_risk) > self.max_open_risk:
                self.last_rejection_reason = "MAX_OPEN_RISK"
                self.logger.warning(
                    "Signal rejected: Max open risk exceeded",
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    correlation_id=getattr(signal, "correlation_id", "") or None,
                    regime=getattr(signal.regime, "value", str(signal.regime)),
                    open_risk=open_risk,
                    proposed_risk=proposed_risk,
                    limit=self.max_open_risk,
                    reason_code=self.last_rejection_reason,
                )
                return []

        # 4. Create Order Intent
        risk_cfg2 = (
            self.config.get("risk")
            if isinstance(self.config.get("risk"), dict)
            else None
        ) or {}
        time_in_force = str(risk_cfg2.get("time_in_force", "day") or "day")
        intent = OrderIntent(
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            order_type=OrderType.LIMIT,  # Default to limit for safety
            limit_price=signal.entry_price,
            time_in_force=time_in_force,
            correlation_id=signal.correlation_id,
            stop_loss=signal.stop_price,
            take_profit=signal.target_price,
            strategy=signal.strategy,
            meta={"created_at": signal.generated_at.isoformat()},
        )

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
