from typing import Any, Dict, List, Optional

from src.core.domain import MarketState, OrderIntent, OrderType, Signal, SymbolState
from src.core.logger import StructuredLogger


class RiskManager:
    """
    Enforces risk limits and converts Signals to OrderIntents.
    """

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.max_daily_loss = config.get("max_daily_loss", 1000.0)
        self.max_risk_per_trade = config.get("max_risk_per_trade", 50.0)  # In dollars
        self.max_orders_per_day = config.get("max_orders_per_day", 100)
        self.max_open_positions = config.get("max_open_positions", 5)
        self.max_notional_per_order = config.get("max_notional_per_order", 5000.0)

        self.current_daily_pnl = 0.0
        self.daily_order_count = 0

    def apply(
        self,
        signal: Signal,
        symbol_state: SymbolState,
        _market_state: MarketState,
        current_positions: Optional[List[Any]] = None,
    ) -> Optional[OrderIntent]:
        """
        Evaluates a signal and returns an OrderIntent if approved, or None if rejected.
        """
        # 0. Check Order Count Limit
        if self.daily_order_count >= self.max_orders_per_day:
            self.logger.warning(
                "Signal rejected: Max daily orders exceeded",
                count=self.daily_order_count,
                limit=self.max_orders_per_day,
                reason_code="MAX_DAILY_ORDERS",
            )
            return None

        # 0b. Check Global Position Limit
        # Only relevant for NEW entries (not exits or reductions)
        # Assuming signal.side matches entry direction (e.g. BUY for LONG entry)
        # We need to know if this creates a NEW position or adds to existing.
        # If symbol_state.position is None, it's a new position.
        if current_positions is not None and symbol_state.position is None:
            if len(current_positions) >= self.max_open_positions:
                self.logger.warning(
                    "Signal rejected: Max open positions reached",
                    current=len(current_positions),
                    limit=self.max_open_positions,
                    reason_code="MAX_POSITIONS",
                )
                return None

            # 0c. Check Strategy Position Limit
            max_strat_pos = self.config.get("max_positions_per_strategy", 3)
            strat_positions = [
                p
                for p in current_positions
                if getattr(p, "strategy", "") == signal.strategy
            ]
            if len(strat_positions) >= max_strat_pos:
                self.logger.warning(
                    "Signal rejected: Max positions for strategy reached",
                    strategy=signal.strategy,
                    current=len(strat_positions),
                    limit=max_strat_pos,
                    reason_code="MAX_STRAT_POSITIONS",
                )
                return None

        # 1. Check Daily Loss Limit
        if self.current_daily_pnl <= -self.max_daily_loss:
            self.logger.warning(
                "Signal rejected: Max daily loss exceeded",
                current_pnl=self.current_daily_pnl,
                limit=self.max_daily_loss,
                reason_code="MAX_DAILY_LOSS",
            )
            return None

        # 1b. Check Open Positions Limit (Approximate)
        # Note: limiting new entries only.
        # This requires accurate symbol_state.position tracking or a global position counter.
        # Since we don't have a global position count passed here, we might need to rely on the engine or query it?
        # Ideally RiskManager should know current total positions.
        # For now, we'll skip global position check here unless we inject it, OR we assume symbol_state has it?
        # Wait, the instruction said "Add strict risk limits".
        # Let's assume we can't easily check GLOBAL positions here without more context.
        # BUT, we can check if WE are already in a position for THIS symbol.

        # To strictly enforce MAX OPEN POSITIONS, the engine needs to pass that info or RiskManager needs to track it.
        # Let's verify if we can access it.
        # For this specific task, let's implement the other checks first and maybe a placeholder/comment for global positions if accessible.

        # 2. Calculate Position Size based on Risk
        # Risk = |Entry - Stop| * Qty
        # Qty = MaxRisk / |Entry - Stop|

        risk_per_share = abs(signal.entry_price - signal.stop_price)
        if risk_per_share <= 0:
            self.logger.warning(
                "Signal rejected: Invalid stop price (zero risk per share)",
                signal=signal,
                reason_code="INVALID_STOP",
            )
            return None

        qty_limit = self.max_risk_per_trade / risk_per_share
        qty_limit = int(qty_limit)  # Floor to be safe

        # If signal provides a size hint, respect it UP TO the limit
        if signal.size_hint:
            qty = min(int(signal.size_hint), qty_limit)
        else:
            qty = qty_limit

        if qty <= 0:
            self.logger.warning(
                "Signal rejected: Calculated quantity is zero (Risk Limit exceeded)",
                signal=signal,
                qty_limit=qty_limit,
                risk_per_share=risk_per_share,
                max_risk=self.max_risk_per_trade,
                reason_code="ZERO_QTY",
            )
            return None

        # 3. Check Notional Value
        notional = qty * signal.entry_price
        if notional > self.max_notional_per_order:
            self.logger.warning(
                "Signal rejected: Max notional per order exceeded",
                notional=notional,
                limit=self.max_notional_per_order,
                reason_code="MAX_NOTIONAL",
            )
            return None

        # 4. Create Order Intent
        intent = OrderIntent(
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            order_type=OrderType.LIMIT,  # Default to limit for safety
            limit_price=signal.entry_price,
            time_in_force="day",
            correlation_id=signal.correlation_id,
            stop_loss=signal.stop_price,
            take_profit=signal.target_price,
            strategy=signal.strategy,
        )

        self.daily_order_count += 1
        self.logger.info(
            "Signal approved", intent=intent, daily_order_count=self.daily_order_count
        )
        return intent

    def update_pnl(self, pnl: float):
        """
        Updates the current daily PnL.
        """
        self.current_daily_pnl += pnl
