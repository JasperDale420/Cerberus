from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from src.analysis.db import DatabaseDatabase
from src.analysis.schema import Order as DbOrder
from src.core.domain import OrderIntent
from src.core.logger import StructuredLogger
from src.data.client import UnifiedDataClient


def _truncate(value: Any, *, limit: int) -> str:
    s = str(value)
    if len(s) <= limit:
        return s
    return s[:limit] + "...(truncated)"


def _extract_broker_error_payload(exc: Exception) -> dict[str, Any]:
    """
    PRD 11.3: OrderExecutor failures must include broker response payloads when available.

    Best-effort extraction for common SDK exception shapes. All large payload fields are
    deterministically truncated to keep logs and DB rows bounded.
    """
    payload: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "error": str(exc),
    }

    for attr in ("status_code", "code", "message", "request_id"):
        try:
            v = getattr(exc, attr, None)
        except Exception:
            v = None
        if v is not None:
            payload[attr] = v

    try:
        resp = getattr(exc, "response", None)
    except Exception:
        resp = None

    if resp is not None:
        try:
            payload["response_status_code"] = getattr(resp, "status_code", None)
        except Exception:
            pass

        # Common patterns: resp.text is either a string attribute or callable method/property.
        text_val: Any = None
        try:
            text_val = getattr(resp, "text", None)
            if callable(text_val):
                text_val = text_val()
        except Exception:
            text_val = None
        if text_val is not None:
            payload["response_text"] = _truncate(text_val, limit=2000)

        # Some clients expose JSON/body fields.
        try:
            json_val = getattr(resp, "json", None)
            if callable(json_val):
                json_val = json_val()
        except Exception:
            json_val = None
        if json_val is not None:
            payload["response_json"] = _truncate(json_val, limit=2000)

    return payload


class OrderExecutor:
    """
    Handles order submission to Alpaca.

    PRD 6.5/6.6: OCO exits are represented via Alpaca bracket orders. The caller provides a single
    entry `OrderIntent` and sets `stop_loss` and/or `take_profit` to request broker-managed exits.
    """

    broker_managed_exits: bool = True

    def __init__(
        self,
        alpaca_client: Any,
        logger: StructuredLogger,
        db: Optional[DatabaseDatabase] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.alpaca_client = alpaca_client
        self.logger = logger
        self.db = db
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))

    def configure_advanced_exits(self, risk_cfg: dict) -> None:
        """
        Configure advanced exits mode from risk config.

        When advanced_exits.enabled is True, disables broker_managed_exits
        so PositionManager handles trailing stops and partial exits.
        """
        adv_exits = risk_cfg.get("advanced_exits", {})
        if isinstance(adv_exits, dict) and adv_exits.get("enabled", False):
            self.broker_managed_exits = False
            self.logger.info(
                "Advanced exits enabled, broker_managed_exits disabled",
                trailing_stop=adv_exits.get("trailing_stop", {}).get("enabled", False),
                partial_exits=adv_exits.get("partial_exits", {}).get("enabled", False),
                regime_aware_stops=adv_exits.get("regime_aware_stops", True),
            )

    def submit(self, intent: OrderIntent):
        """
        Submits an order to Alpaca based on OrderIntent.
        """
        _bind = getattr(self.logger, "bind", None)
        log = (
            _bind(
                symbol=intent.symbol,
                strategy=intent.strategy,
                correlation_id=intent.correlation_id,
            )
            if callable(_bind)
            else self.logger
        )
        try:

            def _map_time_in_force(value: str) -> TimeInForce:
                v = str(value or "").strip().lower()
                if v in ("day", "d"):
                    return TimeInForce.DAY
                if v in ("gtc", "good_till_cancel", "good-til-cancel"):
                    return TimeInForce.GTC
                if v in ("ioc", "immediate_or_cancel", "immediate-or-cancel"):
                    return TimeInForce.IOC
                if v in ("fok", "fill_or_kill", "fill-or-kill"):
                    return TimeInForce.FOK
                return TimeInForce.DAY

            side = OrderSide.BUY if intent.side.value == "buy" else OrderSide.SELL
            time_in_force = _map_time_in_force(getattr(intent, "time_in_force", "day"))

            # Construct Bracket Order if SL/TP are present AND broker manages exits
            # When advanced_exits is enabled, we skip bracket orders and let PositionManager handle exits
            tp_req = None
            sl_req = None
            if self.broker_managed_exits:
                if intent.take_profit:
                    tp_req = TakeProfitRequest(limit_price=intent.take_profit)
                if intent.stop_loss:
                    sl_req = StopLossRequest(stop_price=intent.stop_loss)

            req: MarketOrderRequest | LimitOrderRequest
            if intent.order_type.value == "market":
                req = MarketOrderRequest(
                    symbol=intent.symbol,
                    qty=intent.qty,
                    side=side,
                    time_in_force=time_in_force,
                    client_order_id=intent.correlation_id,
                    take_profit=tp_req,
                    stop_loss=sl_req,
                )
            else:
                req = LimitOrderRequest(
                    symbol=intent.symbol,
                    qty=intent.qty,
                    side=side,
                    time_in_force=time_in_force,
                    limit_price=intent.limit_price,
                    client_order_id=intent.correlation_id,
                    take_profit=tp_req,
                    stop_loss=sl_req,
                )

            order = self.alpaca_client.trading_client.submit_order(req)
            order_id = str(getattr(order, "id", order))

            log.info(
                "Order submitted",
                order_id=order_id,
                symbol=intent.symbol,
                correlation_id=intent.correlation_id,
                broker_payload=getattr(order, "__dict__", None) or str(order),
            )

            # Persist Order
            if self.db:
                created_at = None
                if isinstance(intent.meta, dict):
                    created_at = intent.meta.get("created_at")
                placed = (
                    datetime.fromisoformat(created_at) if isinstance(created_at, str) and created_at else self.clock()
                )
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
                            status="submitted",  # Initial status
                            time_placed=placed,
                            time_last_update=placed,
                            broker_order_id=order_id,
                            meta_json={
                                **(intent.meta or {}),
                                "broker_order_id": order_id,
                                "strategy": intent.strategy,
                                "stop_loss": intent.stop_loss,
                                "take_profit": intent.take_profit,
                                "time_in_force": intent.time_in_force,
                            },
                        )
                    ),
                )

            return order

        except Exception as e:
            from src.core.errors import ErrorCode

            error = str(e)
            broker_error_payload = _extract_broker_error_payload(e)
            log.error(
                "Order submission failed",
                error_code=ErrorCode.ORDER_SUBMIT_FAILED.value,
                symbol=intent.symbol,
                correlation_id=intent.correlation_id,
                error=error,
                exception_type=type(e).__name__,
                broker_error_payload=broker_error_payload,
                intent={
                    "symbol": intent.symbol,
                    "side": intent.side.value,
                    "qty": intent.qty,
                    "order_type": intent.order_type.value,
                    "limit_price": intent.limit_price,
                    "time_in_force": intent.time_in_force,
                    "strategy": intent.strategy,
                    "stop_loss": intent.stop_loss,
                    "take_profit": intent.take_profit,
                },
                exc_info=True,
            )

            # Persist failed order
            if self.db:
                # P3.2 fix: Capture clock once to ensure consistent timestamps
                fail_time = self.clock()
                self.db.write(
                    "order_failed",
                    lambda session: session.add(
                        DbOrder(
                            correlation_id=intent.correlation_id,
                            symbol=intent.symbol,
                            side=intent.side.value,
                            qty=intent.qty,
                            type=intent.order_type.value,
                            limit_price=intent.limit_price,
                            status="order_failed",
                            time_placed=fail_time,
                            time_last_update=fail_time,
                            broker_order_id=None,
                            meta_json={
                                "error": error,
                                "correlation_id": intent.correlation_id,
                                "broker_error_payload": broker_error_payload,
                                **(intent.meta or {}),
                            },
                        )
                    ),
                )

            raise

    def cancel_by_broker_order_id(self, broker_order_id: str) -> None:
        """
        Best-effort cancel of a specific Alpaca order by broker order ID.
        """
        try:
            self.alpaca_client.trading_client.cancel_order_by_id(broker_order_id)
            self.logger.info("Order cancel requested", broker_order_id=broker_order_id)
        except Exception as e:
            self.logger.error(
                "Order cancel failed",
                broker_order_id=broker_order_id,
                error=str(e),
                exc_info=True,
            )
            raise

    def cancel_all_for_symbol(self, symbol: str) -> int:
        """Cancel all open orders for a symbol. Returns count of cancels attempted."""
        cancelled = 0
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            resp = self.alpaca_client.trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
            orders = resp if isinstance(resp, list) else []
            for o in orders:
                if getattr(o, "symbol", "") == symbol:
                    self.cancel_by_broker_order_id(str(getattr(o, "id", "")))
                    cancelled += 1
        except Exception as e:
            # P2.1 fix: Log but also return count so caller knows what happened
            self.logger.error(
                "Cancel-all for symbol failed",
                symbol=symbol,
                cancelled_before_error=cancelled,
                error=str(e),
                exc_info=True,
            )
        return cancelled

    def handle_trade_update(self, update: object) -> dict:
        """
        PRD 6.6: handle order status updates and map broker order IDs back to correlation IDs.

        Returns a normalized dict with fields used by the engine for fill handling:
        - event, timestamp, symbol, side, status, broker_order_id, correlation_id, strategy
        - fill_qty, fill_price (only for fill/partial_fill events)
        """
        try:
            event_raw = getattr(update, "event", None)
            event = getattr(event_raw, "value", None) or str(event_raw or "")
            ts = getattr(update, "timestamp", None)
            order = getattr(update, "order", None)
            broker_order_id = str(getattr(order, "id", "") or "")
            symbol = str(getattr(order, "symbol", "") or "")
            client_order_id = str(getattr(order, "client_order_id", "") or "")
            parent_broker_order_id = str(getattr(order, "parent_order_id", "") or getattr(order, "parent_id", "") or "")
            side_raw = getattr(order, "side", None)
            side = getattr(side_raw, "value", None) or str(side_raw or "")
            status_raw = getattr(order, "status", None)
            status = getattr(status_raw, "value", None) or str(status_raw or "")
        except Exception as e:
            self.logger.error("Trade update parse failed", error=str(e))
            return {}

        correlation_id: str = client_order_id or ""
        # PRD 2.1/3.2: ensure we can always return a non-empty correlation_id for tracing,
        # even when broker events omit client_order_id (e.g., bracket child orders).
        correlation_id_fallback: str = ""
        if not correlation_id:
            if broker_order_id:
                correlation_id_fallback = f"alpaca-{broker_order_id}"
            elif parent_broker_order_id:
                correlation_id_fallback = f"alpaca-{parent_broker_order_id}"
            else:
                seed = f"{symbol}|{event}|{ts}"
                digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
                correlation_id_fallback = f"alpaca-{digest}"
        strategy_name: str = ""

        # Persist order status updates and retrieve correlation_id/strategy for mapping.
        # Also upsert missing rows for orders created outside submit-time DB persistence (e.g., bracket legs).
        if self.db and broker_order_id:
            now = self.clock()

            def _update_order(session) -> None:
                nonlocal correlation_id, correlation_id_fallback, strategy_name
                # PRD 6.6: best-effort mapping of broker order IDs to correlation IDs.
                # Bracket/OTO orders may emit child orders without `client_order_id`; try to inherit
                # correlation_id + strategy from the parent broker_order_id when available.
                if (not correlation_id) and parent_broker_order_id:
                    parent = (
                        session.query(DbOrder)
                        .filter(DbOrder.broker_order_id == parent_broker_order_id)
                        .order_by(DbOrder.id.desc())
                        .first()
                    )
                    if parent is not None:
                        correlation_id = str(parent.correlation_id or "") or correlation_id
                        pmeta = parent.meta_json or {}
                        strategy_name = str(pmeta.get("strategy", "") or "") or strategy_name

                row = (
                    session.query(DbOrder)
                    .filter(DbOrder.broker_order_id == broker_order_id)
                    .order_by(DbOrder.id.desc())
                    .first()
                )
                if row is None:
                    corr_to_use = correlation_id or correlation_id_fallback
                    inferred_type = getattr(getattr(order, "type", None), "value", None) or str(
                        getattr(order, "type", "") or ""
                    )
                    inferred_qty = float(getattr(order, "qty", getattr(order, "quantity", 0.0)) or 0.0)
                    inferred_limit = getattr(order, "limit_price", None)
                    placed = getattr(order, "created_at", None) or now

                    session.add(
                        DbOrder(
                            correlation_id=corr_to_use,
                            symbol=symbol or "unknown",
                            side=side or "unknown",
                            qty=inferred_qty,
                            type=inferred_type or "unknown",
                            limit_price=(float(inferred_limit) if inferred_limit is not None else None),
                            status=status or "unknown",
                            time_placed=placed,
                            time_last_update=ts or now,
                            broker_order_id=broker_order_id,
                            meta_json={
                                "broker_order_id": broker_order_id,
                                "client_order_id": client_order_id,
                                "parent_broker_order_id": parent_broker_order_id or None,
                                "recovered_from_trade_stream": True,
                            },
                        )
                    )
                    return

                correlation_id = str(row.correlation_id or "") or correlation_id
                meta = row.meta_json or {}
                strategy_name = str(meta.get("strategy", "") or "")

                if status:
                    row.status = status
                row.time_last_update = ts or now
                meta.update(
                    {
                        "last_trade_event": event,
                        "broker_status": status,
                        "trade_update_ts": str(ts or now),
                        "parent_broker_order_id": parent_broker_order_id or meta.get("parent_broker_order_id"),
                    }
                )
                row.meta_json = meta

            self.db.write("order_update", _update_order)

        correlation_id = correlation_id or correlation_id_fallback

        out: dict = {
            "event": event,
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "status": status,
            "broker_order_id": broker_order_id or None,
            "client_order_id": client_order_id or None,
            "parent_broker_order_id": parent_broker_order_id or None,
            "correlation_id": correlation_id or None,
            "strategy": strategy_name or None,
        }

        if event in ("fill", "partial_fill"):
            try:
                qty = float(getattr(update, "qty", 0.0) or 0.0)
                price = float(getattr(update, "price", 0.0) or 0.0)
            except Exception:
                qty = 0.0
                price = 0.0
            out["fill_qty"] = qty
            out["fill_price"] = price

        return out


class GatewayOrderExecutor:
    """Order executor that routes trading operations through Data-Gateway."""

    broker_managed_exits: bool = False

    def __init__(
        self,
        unified_client: UnifiedDataClient,
        logger: StructuredLogger,
        db: Optional[DatabaseDatabase] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.unified_client = unified_client
        self.logger = logger
        self.db = db
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))

    def configure_advanced_exits(self, risk_cfg: dict) -> None:
        """Gateway execution uses local exit management; keep broker exits disabled."""
        self.broker_managed_exits = False
        self.logger.info("Gateway order executor configured", broker_managed_exits=False)

    def submit(self, intent: OrderIntent) -> dict[str, Any]:
        """Submit an order via Data-Gateway Alpaca trading endpoint."""
        _bind = getattr(self.logger, "bind", None)
        log = (
            _bind(
                symbol=intent.symbol,
                strategy=intent.strategy,
                correlation_id=intent.correlation_id,
            )
            if callable(_bind)
            else self.logger
        )
        try:
            order = self.unified_client.submit_order(
                symbol=intent.symbol,
                side=intent.side.value,
                qty=float(intent.qty),
                order_type=intent.order_type.value,
                time_in_force=intent.time_in_force,
                limit_price=float(intent.limit_price) if intent.limit_price is not None else None,
                client_order_id=intent.correlation_id,
            )
            order_id = str(order.get("id") or order.get("order_id") or "")

            log.info(
                "Order submitted via gateway",
                order_id=order_id or None,
                symbol=intent.symbol,
                correlation_id=intent.correlation_id,
                broker_payload=order,
            )

            if self.db:
                created_at = None
                if isinstance(intent.meta, dict):
                    created_at = intent.meta.get("created_at")
                placed = (
                    datetime.fromisoformat(created_at) if isinstance(created_at, str) and created_at else self.clock()
                )
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
                            status="submitted",
                            time_placed=placed,
                            time_last_update=placed,
                            broker_order_id=order_id or None,
                            meta_json={
                                **(intent.meta or {}),
                                "broker_order_id": order_id or None,
                                "strategy": intent.strategy,
                                "stop_loss": intent.stop_loss,
                                "take_profit": intent.take_profit,
                                "time_in_force": intent.time_in_force,
                                "execution_backend": "gateway",
                            },
                        )
                    ),
                )

            return order
        except Exception as e:
            from src.core.errors import ErrorCode

            error = str(e)
            broker_error_payload = _extract_broker_error_payload(e)
            log.error(
                "Gateway order submission failed",
                error_code=ErrorCode.ORDER_SUBMIT_FAILED.value,
                symbol=intent.symbol,
                correlation_id=intent.correlation_id,
                error=error,
                exception_type=type(e).__name__,
                broker_error_payload=broker_error_payload,
                intent={
                    "symbol": intent.symbol,
                    "side": intent.side.value,
                    "qty": intent.qty,
                    "order_type": intent.order_type.value,
                    "limit_price": intent.limit_price,
                    "time_in_force": intent.time_in_force,
                    "strategy": intent.strategy,
                    "stop_loss": intent.stop_loss,
                    "take_profit": intent.take_profit,
                },
                exc_info=True,
            )

            if self.db:
                fail_time = self.clock()
                self.db.write(
                    "order_failed",
                    lambda session: session.add(
                        DbOrder(
                            correlation_id=intent.correlation_id,
                            symbol=intent.symbol,
                            side=intent.side.value,
                            qty=intent.qty,
                            type=intent.order_type.value,
                            limit_price=intent.limit_price,
                            status="order_failed",
                            time_placed=fail_time,
                            time_last_update=fail_time,
                            broker_order_id=None,
                            meta_json={
                                "error": error,
                                "correlation_id": intent.correlation_id,
                                "broker_error_payload": broker_error_payload,
                                "execution_backend": "gateway",
                                **(intent.meta or {}),
                            },
                        )
                    ),
                )

            raise

    def cancel_by_broker_order_id(self, broker_order_id: str) -> None:
        """Best-effort cancel of a specific order through Data-Gateway."""
        try:
            cancelled = self.unified_client.cancel_order(broker_order_id)
            self.logger.info(
                "Gateway order cancel requested",
                broker_order_id=broker_order_id,
                cancelled=bool(cancelled),
            )
        except Exception as e:
            self.logger.error(
                "Gateway order cancel failed",
                broker_order_id=broker_order_id,
                error=str(e),
                exc_info=True,
            )
            raise

    def cancel_all_for_symbol(self, symbol: str) -> int:
        """Cancel all open orders for a symbol via Data-Gateway."""
        cancelled = 0
        try:
            orders = self.unified_client.get_orders(status="open", limit=500)
            for order in orders:
                if str(order.get("symbol", "")) != symbol:
                    continue
                oid = str(order.get("id") or "")
                if not oid:
                    continue
                self.cancel_by_broker_order_id(oid)
                cancelled += 1
        except Exception as e:
            self.logger.error(
                "Gateway cancel-all for symbol failed",
                symbol=symbol,
                cancelled_before_error=cancelled,
                error=str(e),
                exc_info=True,
            )
        return cancelled

    def handle_trade_update(self, update: object) -> dict:
        """Trade updates are currently handled via reconciliation when using gateway execution."""
        return {}


class NoopOrderExecutor:
    """
    PRD 10.2: OrderExecutor stub / paper trading mode.

    This executor never calls a broker. It deterministically logs and (optionally)
    persists simulated orders for end-to-end vertical-slice testing.
    """

    def __init__(
        self,
        logger: StructuredLogger,
        db: Optional[DatabaseDatabase] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.logger = logger
        self.db = db
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))
        self.broker_managed_exits: bool = False

    def submit(self, intent: OrderIntent) -> dict[str, Any]:
        now = self.clock()
        broker_order_id = f"noop-{intent.correlation_id}"
        self.logger.info(
            "Noop order submitted",
            symbol=intent.symbol,
            strategy=intent.strategy,
            correlation_id=intent.correlation_id,
            broker_order_id=broker_order_id,
            intent=intent,
        )

        if self.db:

            def _write(session) -> None:
                session.add(
                    DbOrder(
                        correlation_id=intent.correlation_id,
                        trade_id=None,
                        symbol=intent.symbol,
                        side=intent.side.value,
                        qty=float(intent.qty),
                        type=intent.order_type.value,
                        limit_price=(float(intent.limit_price) if intent.limit_price is not None else None),
                        status="simulated",
                        time_placed=now,
                        time_last_update=now,
                        broker_order_id=broker_order_id,
                        meta_json={
                            **(intent.meta or {}),
                            "simulated": True,
                            "strategy": intent.strategy,
                            "stop_loss": intent.stop_loss,
                            "take_profit": intent.take_profit,
                        },
                    )
                )

            self.db.write("order", _write)

        return {"id": broker_order_id, "status": "simulated"}

    def cancel_by_broker_order_id(self, broker_order_id: str) -> None:
        self.logger.info("Noop order cancel requested", broker_order_id=str(broker_order_id))

    def cancel_all_for_symbol(self, symbol: str) -> int:
        self.logger.info("Noop cancel-all for symbol requested", symbol=str(symbol))
        return 0

    def handle_trade_update(self, update: object) -> dict:
        # No broker trade updates exist in noop mode.
        return {}
