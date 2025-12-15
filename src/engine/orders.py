from datetime import datetime, timezone
from typing import Optional

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
from src.data.alpaca import AlpacaClient


class OrderExecutor:
    """
    Handles order submission to Alpaca.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        logger: StructuredLogger,
        db: Optional[DatabaseDatabase] = None,
    ):
        self.alpaca_client = alpaca_client
        self.logger = logger
        self.db = db

    def submit(self, intent: OrderIntent):
        """
        Submits an order to Alpaca based on OrderIntent.
        """
        try:
            side = OrderSide.BUY if intent.side.value == "buy" else OrderSide.SELL

            # Construct Bracket Order if SL/TP are present
            # Alpaca API supports bracket orders via take_profit and stop_loss params

            tp_req = None
            if intent.take_profit:
                tp_req = TakeProfitRequest(limit_price=intent.take_profit)

            sl_req = None
            if intent.stop_loss:
                sl_req = StopLossRequest(stop_price=intent.stop_loss)

            req: MarketOrderRequest | LimitOrderRequest
            if intent.order_type.value == "market":
                req = MarketOrderRequest(
                    symbol=intent.symbol,
                    qty=intent.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    take_profit=tp_req,
                    stop_loss=sl_req,
                )
            else:
                req = LimitOrderRequest(
                    symbol=intent.symbol,
                    qty=intent.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=intent.limit_price,
                    take_profit=tp_req,
                    stop_loss=sl_req,
                )

            order = self.alpaca_client.trading_client.submit_order(req)
            order_id = str(getattr(order, "id", order))

            self.logger.info(
                "Order submitted",
                order_id=order_id,
                symbol=intent.symbol,
            )

            # Persist Order
            if self.db:
                try:
                    with self.db.get_session() as session:
                        db_order = DbOrder(
                            correlation_id=intent.correlation_id,
                            symbol=intent.symbol,
                            side=intent.side.value,
                            qty=intent.qty,
                            type=intent.order_type.value,
                            limit_price=intent.limit_price,
                            status="submitted",  # Initial status
                            time_placed=datetime.now(timezone.utc),
                            time_last_update=datetime.now(timezone.utc),
                            broker_order_id=order_id,
                            meta_json=intent.meta,
                        )
                        session.add(db_order)
                except Exception as e:
                    self.logger.error("Failed to persist order", error=str(e))

            return order

        except Exception as e:
            self.logger.error(
                "Order submission failed", symbol=intent.symbol, error=str(e)
            )

            # Persist failed order
            if self.db:
                try:
                    with self.db.get_session() as session:
                        db_order = DbOrder(
                            correlation_id=intent.correlation_id,
                            symbol=intent.symbol,
                            side=intent.side.value,
                            qty=intent.qty,
                            type=intent.order_type.value,
                            limit_price=intent.limit_price,
                            status="order_failed",
                            time_placed=datetime.now(timezone.utc),
                            time_last_update=datetime.now(timezone.utc),
                            broker_order_id=None,
                            meta_json={"error": str(e), **intent.meta},
                        )
                        session.add(db_order)
                except Exception as db_e:
                    self.logger.error("Failed to persist failed order", error=str(db_e))

            raise
