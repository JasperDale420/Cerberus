import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.core.domain import OrderIntent
from src.core.logger import StructuredLogger


class SimulatedOrderExecutor:
    """
    Simulated order executor for production-code backtesting.
    Mocks Alpaca trading client behavior, maintaining internal open orders,
    and triggering simulated fills when process_bars is called with new market data.
    """

    broker_managed_exits: bool = True

    def __init__(
        self,
        logger: StructuredLogger,
        db: Optional[DatabaseDatabase] = None,
        clock: Optional[Callable[[], datetime]] = None,
        on_trade_update: Optional[Callable[[Any], None]] = None,
        account: Optional[Any] = None,
        backtest_cfg: Optional[Dict[str, Any]] = None,
        risk_cfg: Optional[Dict[str, Any]] = None,
        fill_model: Optional[Any] = None,
    ):
        self.logger = logger
        self.db = db
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.on_trade_update = on_trade_update
        self.account = account
        self.fill_model = fill_model

        # Slippage & cost config
        rk = risk_cfg or {}
        self._slippage_bps: float = float(rk.get("slippage_bps", 0.0) or 0.0)
        self._commission_per_share: float = float(rk.get("commission_per_share", 0.0) or 0.0)
        self._min_commission: float = float(rk.get("min_commission", 0.0) or 0.0)

        # Internal state
        self.open_orders: Dict[str, Dict[str, Any]] = {}

    def configure_advanced_exits(self, risk_cfg: dict) -> None:
        adv_exits = risk_cfg.get("advanced_exits", {})
        if isinstance(adv_exits, dict) and adv_exits.get("enabled", False):
            self.broker_managed_exits = False

    def submit(self, intent: OrderIntent) -> Dict[str, Any]:
        order_id = f"sim-{uuid.uuid4().hex[:8]}"
        now = self.clock()

        order = {
            "id": order_id,
            "symbol": intent.symbol,
            "side": intent.side.value,
            "qty": intent.qty,
            "type": intent.order_type.value,
            "limit_price": intent.limit_price,
            "time_in_force": intent.time_in_force,
            "status": "new",
            "created_at": now,
            "client_order_id": intent.correlation_id,
            "take_profit": intent.take_profit,
            "stop_loss": intent.stop_loss,
            "strategy": intent.strategy,
            "correlation_id": intent.correlation_id,
        }

        self.open_orders[order_id] = order

        if self.db:
            created_at = None
            if isinstance(intent.meta, dict):
                created_at = intent.meta.get("created_at")
            placed = datetime.fromisoformat(created_at) if isinstance(created_at, str) and created_at else now

            self.db.write(
                "order",
                lambda session: session.add(
                    DbOrder(
                        correlation_id=intent.correlation_id,
                        symbol=intent.symbol,
                        side=intent.side.value,
                        qty=intent.qty,
                        type=intent.order_type.value,
                        limit_price=intent.limit_price,
                        status="new",
                        time_placed=placed,
                        time_last_update=placed,
                        broker_order_id=order_id,
                        meta_json={
                            **(intent.meta or {}),
                            "broker_order_id": order_id,
                            "strategy": intent.strategy,
                            "simulated": True,
                        },
                    )
                ),
            )

        # Immediately notify new order created
        if self.on_trade_update:
            self._dispatch_event("new", order)

        return {"id": order_id, "status": "new"}

    def cancel_by_broker_order_id(self, broker_order_id: str) -> None:
        if broker_order_id in self.open_orders:
            order = self.open_orders.pop(broker_order_id)
            order["status"] = "canceled"
            if self.on_trade_update:
                self._dispatch_event("canceled", order)

    def cancel_all_for_symbol(self, symbol: str) -> int:
        to_cancel = [oid for oid, o in self.open_orders.items() if o["symbol"] == symbol]
        for oid in to_cancel:
            self.cancel_by_broker_order_id(oid)
        return len(to_cancel)

    def handle_trade_update(self, update: object) -> dict:
        # In simulated mode, `update` is already the normalized dict
        if isinstance(update, dict):
            return update
        # Handle _MockUpdate objects from _dispatch_event — extract attributes
        # so ExecutionEngine.on_trade_update can process fills into PositionManager
        if hasattr(update, "event"):
            result: dict = {}
            for attr in (
                "event",
                "timestamp",
                "symbol",
                "side",
                "status",
                "broker_order_id",
                "client_order_id",
                "parent_broker_order_id",
                "correlation_id",
                "strategy",
                "qty",
                "price",
                "fill_qty",
                "fill_price",
            ):
                val = getattr(update, attr, None)
                if val is not None:
                    # Unwrap mock enums
                    if hasattr(val, "value"):
                        val = val.value
                    result[attr] = val
            # on_trade_update expects fill_qty/fill_price for fill events
            if "fill_qty" not in result and "qty" in result:
                result["fill_qty"] = result["qty"]
            if "fill_price" not in result and "price" in result:
                result["fill_price"] = result["price"]
            return result
        return {}

    def _dispatch_event(
        self, event_type: str, order: Dict[str, Any], fill_qty: float = 0.0, fill_price: float = 0.0
    ) -> None:
        if not self.on_trade_update:
            return

        event_payload = {
            "event": event_type,
            "timestamp": self.clock(),
            "symbol": order["symbol"],
            "side": order["side"],
            "status": order["status"],
            "broker_order_id": order["id"],
            "client_order_id": order.get("client_order_id"),
            "parent_broker_order_id": order.get("parent_broker_order_id"),
            "correlation_id": order.get("correlation_id"),
            "strategy": order.get("strategy"),
        }

        if event_type in ("fill", "partial_fill"):
            event_payload["fill_qty"] = fill_qty
            event_payload["fill_price"] = fill_price

            if self.db:
                self.db.write(
                    "order_fill",
                    lambda session: session.query(DbOrder)
                    .filter(DbOrder.broker_order_id == order["id"])
                    .update(
                        {
                            "status": "filled",
                            "limit_price": fill_price,
                            "time_last_update": self.clock(),
                        }
                    ),
                )

        import asyncio

        # ExecutionEngine expects on_trade_update to be a method that takes Alpaca's object.
        # However, our execution.py wraps and extracts, so simulated needs to yield simple objects that behave that way.
        class _MockUpdate:
            pass

        mock = _MockUpdate()
        for k, v in event_payload.items():
            setattr(mock, k, v)

        # Mock enum-like properties
        class _MockEnum:
            def __init__(self, val):
                self.value = val

        class _MockOrder:
            def __init__(self, o_dict):
                self.id = o_dict["id"]
                self.symbol = o_dict["symbol"]
                self.client_order_id = o_dict.get("client_order_id", "")
                self.side = _MockEnum(o_dict["side"])
                self.status = _MockEnum(o_dict["status"])
                self.parent_id = o_dict.get("parent_broker_order_id")

        mock.event = _MockEnum(event_type)
        mock.timestamp = event_payload["timestamp"]
        mock.order = _MockOrder(order)
        if event_type in ("fill", "partial_fill"):
            mock.qty = fill_qty
            mock.price = fill_price

        # In backtest mode, dispatch fills synchronously so PositionManager
        # processes them immediately (not deferred via create_task which
        # only executes on the next event loop yield).
        # Schedule as a task — the runner must await asyncio.sleep(0) after
        # each process_bar call to flush these fills before the next bar.
        if asyncio.iscoroutinefunction(self.on_trade_update):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.on_trade_update(mock))
            except RuntimeError:
                pass
        else:
            self.on_trade_update(mock)

    def _apply_slippage(self, price: float, side: str) -> float:
        """Worsen fill price by slippage_bps. Buys pay more, sells receive less."""
        if self._slippage_bps <= 0.0 or price <= 0.0:
            return price
        slip_frac = self._slippage_bps / 10_000.0
        if side == "buy":
            return price * (1.0 + slip_frac)
        return price * (1.0 - slip_frac)

    def _deduct_commission(self, qty: float) -> float:
        """Calculate and deduct commission from account cash. Returns commission amount."""
        commission = max(self._min_commission, self._commission_per_share * qty)
        if commission > 0.0 and self.account is not None and hasattr(self.account, "cash"):
            self.account.cash -= commission
        return commission

    def process_bar(self, bar: Any) -> None:
        """
        Evaluate pending orders against the current bar.
        Market orders fill immediately at bar open (with slippage).
        Limit orders fill if bar low <= limit (buy) or bar high >= limit (sell).
        Bracket orders (stop loss / take profit) are evaluated if parent is filled.
        """
        symbol = getattr(bar, "symbol", getattr(bar, "S", None))
        if not symbol:
            return

        open_p = getattr(bar, "open", 0.0)
        high_p = getattr(bar, "high", 0.0)
        low_p = getattr(bar, "low", 0.0)

        filled_order_ids = []
        # Track parent IDs whose OCO leg already filled this bar to prevent
        # both TP and SL filling on the same bar (OCO double-fill bug).
        filled_parent_ids: set[str] = set()

        for oid, order in list(self.open_orders.items()):
            if order["symbol"] != symbol:
                continue

            # OCO guard: if a sibling with the same parent already filled, skip
            parent_id = order.get("parent_broker_order_id")
            if parent_id and parent_id in filled_parent_ids:
                continue

            fill_price = 0.0
            is_filled = False

            if order["type"] == "market":
                fill_price = self._apply_slippage(open_p, order["side"])
                is_filled = True
            elif order["type"] == "limit":
                limit = float(order["limit_price"])
                if order["side"] == "buy" and low_p <= limit:
                    fill_price = self._apply_slippage(min(limit, open_p), order["side"])
                    is_filled = True
                elif order["side"] == "sell" and high_p >= limit:
                    fill_price = self._apply_slippage(max(limit, open_p), order["side"])
                    is_filled = True
            elif order["type"] == "stop":
                stop = float(order.get("stop_price", 0.0))
                if order["side"] == "sell" and low_p <= stop:
                    fill_price = self._apply_slippage(min(stop, open_p), order["side"])
                    is_filled = True
                elif order["side"] == "buy" and high_p >= stop:
                    fill_price = self._apply_slippage(max(stop, open_p), order["side"])
                    is_filled = True

            if is_filled:
                order["status"] = "filled"

                # Mark parent as having a filled child (OCO guard)
                if parent_id:
                    filled_parent_ids.add(parent_id)

                fill_qty = float(order["qty"])
                fill_val = fill_qty * fill_price

                # Deduct commission from account
                self._deduct_commission(fill_qty)

                if self.account is not None:
                    if hasattr(self.account, "cash") and hasattr(self.account, "positions_qty"):
                        if order["side"] == "buy":
                            self.account.cash -= fill_val
                            self.account.positions_qty[symbol] = self.account.positions_qty.get(symbol, 0.0) + fill_qty
                        else:
                            self.account.cash += fill_val
                            self.account.positions_qty[symbol] = self.account.positions_qty.get(symbol, 0.0) - fill_qty

                self._dispatch_event("fill", order, fill_qty=order["qty"], fill_price=fill_price)
                filled_order_ids.append(oid)

                # Setup OCO children if broker_managed_exits is true
                if self.broker_managed_exits and (order.get("take_profit") or order.get("stop_loss")):
                    if order.get("take_profit"):
                        tp_id = f"sim-{uuid.uuid4().hex[:8]}"
                        tp_side = "sell" if order["side"] == "buy" else "buy"
                        tp_order = {
                            "id": tp_id,
                            "symbol": symbol,
                            "side": tp_side,
                            "qty": order["qty"],
                            "type": "limit",
                            "limit_price": order["take_profit"],
                            "status": "new",
                            "parent_broker_order_id": oid,
                            "strategy": order.get("strategy"),
                            "correlation_id": order.get("correlation_id"),
                        }
                        self.open_orders[tp_id] = tp_order

                        if self.db:
                            from src.analysis.schema import Order as DbOrder

                            placed = self.clock()
                            tp_strat = order.get("strategy")
                            tp_corr = order.get("correlation_id")
                            tp_sym = symbol
                            tp_placed = placed
                            self.db.write(
                                "order",
                                lambda session,
                                t_id=tp_id,
                                t_side=tp_side,
                                t_qty=order["qty"],
                                t_limit=order["take_profit"],
                                t_strat=tp_strat,
                                t_corr=tp_corr,
                                t_sym=tp_sym,
                                t_placed=tp_placed: session.add(
                                    DbOrder(
                                        correlation_id=t_corr,
                                        symbol=t_sym,
                                        side=t_side,
                                        qty=t_qty,
                                        type="limit",
                                        limit_price=t_limit,
                                        status="new",
                                        time_placed=t_placed,
                                        time_last_update=t_placed,
                                        broker_order_id=t_id,
                                        meta_json={
                                            "broker_order_id": t_id,
                                            "strategy": t_strat,
                                            "simulated": True,
                                            "is_exit": True,
                                            "exit_type": "take_profit",
                                        },
                                    )
                                ),
                            )
                        self._dispatch_event("new", tp_order)

                    if order.get("stop_loss"):
                        sl_id = f"sim-{uuid.uuid4().hex[:8]}"
                        sl_side = "sell" if order["side"] == "buy" else "buy"
                        sl_order = {
                            "id": sl_id,
                            "symbol": symbol,
                            "side": sl_side,
                            "qty": order["qty"],
                            "type": "stop",
                            "stop_price": order["stop_loss"],
                            "status": "new",
                            "parent_broker_order_id": oid,
                            "strategy": order.get("strategy"),
                            "correlation_id": order.get("correlation_id"),
                        }
                        self.open_orders[sl_id] = sl_order

                        if self.db:
                            from src.analysis.schema import Order as DbOrder

                            placed = self.clock()
                            sl_strat = order.get("strategy")
                            sl_corr = order.get("correlation_id")
                            sl_sym = symbol
                            sl_placed = placed
                            self.db.write(
                                "order",
                                lambda session,
                                s_id=sl_id,
                                s_side=sl_side,
                                s_qty=order["qty"],
                                s_stop=order["stop_loss"],
                                s_strat=sl_strat,
                                s_corr=sl_corr,
                                s_sym=sl_sym,
                                s_placed=sl_placed: session.add(
                                    DbOrder(
                                        correlation_id=s_corr,
                                        symbol=s_sym,
                                        side=s_side,
                                        qty=s_qty,
                                        type="stop",
                                        limit_price=s_stop,
                                        status="new",
                                        time_placed=s_placed,
                                        time_last_update=s_placed,
                                        broker_order_id=s_id,
                                        meta_json={
                                            "broker_order_id": s_id,
                                            "strategy": s_strat,
                                            "simulated": True,
                                            "is_exit": True,
                                            "exit_type": "stop_loss",
                                            "stop_price": s_stop,
                                        },
                                    )
                                ),
                            )
                        self._dispatch_event("new", sl_order)

        for oid in filled_order_ids:
            # If OCO was triggered, we might need to cancel the other leg. This is a simple backtest, we just pop.
            popped = self.open_orders.pop(oid, None)

            # Cancel siblings (OCO emulation)
            if popped and popped.get("parent_broker_order_id"):
                parent_id = popped["parent_broker_order_id"]
                siblings_to_cancel = [
                    sid
                    for sid, so in self.open_orders.items()
                    if so.get("parent_broker_order_id") == parent_id and sid != oid
                ]
                for sid in siblings_to_cancel:
                    self.cancel_by_broker_order_id(sid)
