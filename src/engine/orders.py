from typing import Optional
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.engine.risk import OrderIntent

class OrderExecutor:
    """
    Handles order submission to Alpaca.
    """
    def __init__(self, alpaca_client: AlpacaClient, logger: StructuredLogger):
        self.alpaca_client = alpaca_client
        self.logger = logger

    def submit(self, intent: OrderIntent):
        """
        Submits an order to Alpaca based on OrderIntent.
        """
        try:
            side = OrderSide.BUY if intent.side == "buy" else OrderSide.SELL
            
            # Construct Bracket Order if SL/TP are present
            # Alpaca API supports bracket orders via take_profit and stop_loss params
            
            tp_req = None
            if intent.take_profit:
                tp_req = TakeProfitRequest(limit_price=intent.take_profit)
                
            sl_req = None
            if intent.stop_loss:
                sl_req = StopLossRequest(stop_price=intent.stop_loss)

            if intent.order_type == "market":
                req = MarketOrderRequest(
                    symbol=intent.symbol,
                    qty=intent.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    take_profit=tp_req,
                    stop_loss=sl_req
                )
            else:
                req = LimitOrderRequest(
                    symbol=intent.symbol,
                    qty=intent.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=intent.limit_price,
                    take_profit=tp_req,
                    stop_loss=sl_req
                )

            order = self.alpaca_client.trading_client.submit_order(req)
            self.logger.info("Order submitted", order_id=str(order.id), symbol=intent.symbol)
            return order

        except Exception as e:
            self.logger.error("Order submission failed", symbol=intent.symbol, error=str(e))
            raise
