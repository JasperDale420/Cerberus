from datetime import datetime
from typing import Any, Dict, Optional

from src.core.logger import StructuredLogger
from src.engine.risk import OrderIntent


class MockOrderExecutor:
    """
    Mocks order execution for backtesting.
    Fills orders immediately at the current price (or next bar open).
    """

    def __init__(self, logger: StructuredLogger, initial_cash: float = 100000.0):
        self.logger = logger
        self.cash = initial_cash
        self.positions: Dict[str, int] = {}  # Symbol -> Quantity
        self.orders: list[Dict[str, Any]] = []
        self.fills: list[Dict[str, Any]] = []

    def submit(self, intent: OrderIntent) -> Optional[Dict]:
        """
        Simulates order submission.
        """
        self.logger.info(
            "Mock submitting order",
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
        )

        # In a simple backtest, we might assume immediate fill at current price
        # But usually we wait for the next bar.
        # For this slice, let's assume immediate fill if we have a 'current_price' context,
        # but the executor doesn't know the price unless passed or looked up.
        # We'll just record the order and let the Runner handle the fill logic
        # OR we assume the intent comes with a price limit/stop or we use the last known price.

        # To keep it simple for the Runner:
        # The Runner will call 'execute_fills' using the next bar's data.

        order_id = f"mock_{len(self.orders) + 1}"
        order = {
            "id": order_id,
            "symbol": intent.symbol,
            "side": intent.side,
            "qty": intent.qty,
            "type": intent.order_type,
            "status": "new",
        }
        self.orders.append(order)
        return order

    def fill_orders(self, symbol: str, price: float, timestamp: datetime):
        """
        Called by the BacktestRunner to fill pending orders at a specific price.
        """
        for order in self.orders:
            if order["symbol"] == symbol and order["status"] == "new":
                # Execute fill
                qty = order["qty"]
                cost = qty * price

                if order["side"] == "buy":
                    self.cash -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + qty
                elif order["side"] == "sell":
                    self.cash += cost
                    self.positions[symbol] = self.positions.get(symbol, 0) - qty

                order["status"] = "filled"
                order["fill_price"] = price
                order["filled_at"] = timestamp

                self.fills.append(order)
                self.logger.info(
                    "Mock order filled", symbol=symbol, side=order["side"], price=price
                )
