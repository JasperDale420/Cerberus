from collections import deque
from typing import Any, Dict, List, Optional


class BacktestAnalyzer:
    """
    Analyzes backtest fills to produce performance metrics.
    matches fills using FIFO (First-In-First-Out) logic to reconstruct round-trip trades.
    """

    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash

    def calculate_statistics(
        self,
        fills: List[Dict[str, Any]],
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Process a list of fills and return a dictionary of statistics.
        Matches trades FIFO and calculates realized + unrealized PnL (if current_prices provided).
        """
        fills = fills or []
        trades, open_positions = self._match_trades(fills)

        if not trades and not open_positions:
            return {
                "total_trades": 0,
                "total_closed_pnl": 0.0,
                "open_pnl": 0.0,
                "total_pnl": 0.0,
                "total_equity": float(self.initial_cash),
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_pnl": 0.0,
                "max_drawdown_pct": 0.0,
                "return_pct": 0.0,
                "trades": [],
                "open_positions": [],
            }

        # --- Realized Metrics (Closed Trades) ---
        pnls = [t["pnl"] for t in trades]
        total_closed_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        # Avoid non-JSON "Infinity" in outputs; treat PF as 0 when denominator is 0.
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        avg_pnl = total_closed_pnl / total_trades if total_trades > 0 else 0.0

        # --- Unrealized Metrics (Open Positions) ---
        open_pnl = 0.0
        valued_open_positions = []

        if current_prices:
            for pos in open_positions:
                symbol = pos["symbol"]
                current_price = current_prices.get(symbol)

                if current_price:
                    if pos["side"] == "buy":  # Long
                        unrealized = (current_price - pos["entry_price"]) * pos["qty"]
                    else:  # Short
                        unrealized = (pos["entry_price"] - current_price) * pos["qty"]

                    open_pnl += unrealized

                    # Enlighten the position record
                    pos["current_price"] = current_price
                    pos["unrealized_pnl"] = round(unrealized, 2)
                    valued_open_positions.append(pos)
                else:
                    # Fallback if no price found (shouldn't happen if runner tracks correctly)
                    pos["unrealized_pnl"] = 0.0
                    valued_open_positions.append(pos)

        total_equity_pnl = total_closed_pnl + open_pnl
        total_equity = self.initial_cash + total_equity_pnl
        return_pct = (total_equity_pnl / self.initial_cash) * 100

        # --- Drawdown Calculation (approximate via closed trades equity curve) ---
        # Note: Proper drawdown needs tick-by-tick equity, but this approximates from realized points
        equity = self.initial_cash
        peak_equity = equity
        max_drawdown = 0.0

        sorted_trades = sorted(trades, key=lambda x: x["exit_time"])

        for trade in sorted_trades:
            equity += trade["pnl"]
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        # If currently in a drawdown due to open positions, account for it
        current_equity = self.initial_cash + total_closed_pnl + open_pnl
        if current_equity > peak_equity:
            peak_equity = current_equity
        current_drawdown = (
            (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
        )
        max_drawdown = max(max_drawdown, current_drawdown)

        return {
            "total_trades": total_trades,
            "total_closed_pnl": round(total_closed_pnl, 2),
            "open_pnl": round(open_pnl, 2),
            "total_pnl": round(total_equity_pnl, 2),
            "total_equity": round(total_equity, 2),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "average_pnl": round(avg_pnl, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "return_pct": round(return_pct, 2),
            "trades": sorted_trades,
            "open_positions": (
                valued_open_positions if current_prices else open_positions
            ),
        }

    def _match_trades(self, fills: List[Dict[str, Any]]) -> Any:
        # Returning Tuple[List[Dict], List[Dict]] - defined as Any to avoid extensive typing imports inline

        # Group fills by symbol
        fills_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for fill in fills:
            sym = fill["symbol"]
            if sym not in fills_by_symbol:
                fills_by_symbol[sym] = []
            fills_by_symbol[sym].append(fill)

        trades = []
        all_open_positions = []

        for symbol, sym_fills in fills_by_symbol.items():
            sym_fills.sort(key=lambda x: x["filled_at"])

            position_stack: deque[Dict[str, Any]] = deque()

            for fill in sym_fills:
                qty = float(fill["qty"])
                price = float(fill["fill_price"])
                side = fill["side"]
                fill_strategy = fill.get("strategy", "unknown")

                remaining_qty = qty

                while remaining_qty > 0.0000001:
                    if not position_stack:
                        # Open new position
                        position_stack.append(
                            {
                                "side": side,
                                "qty": remaining_qty,
                                "price": price,
                                "time": fill["filled_at"],
                                "strategy": fill_strategy,
                            }
                        )
                        remaining_qty = 0
                    else:
                        head = position_stack[0]
                        if head["side"] == side:
                            # Add to position
                            position_stack.append(
                                {
                                    "side": side,
                                    "qty": remaining_qty,
                                    "price": price,
                                    "time": fill["filled_at"],
                                    "strategy": fill_strategy,
                                }
                            )
                            remaining_qty = 0
                        else:
                            # Close position
                            match_qty = min(remaining_qty, head["qty"])

                            entry_price = head["price"]
                            exit_price = price

                            if head["side"] == "buy":
                                pnl = (exit_price - entry_price) * match_qty
                            else:
                                pnl = (entry_price - exit_price) * match_qty

                            trades.append(
                                {
                                    "symbol": symbol,
                                    "side": head["side"],
                                    "entry_time": head["time"],
                                    "exit_time": fill["filled_at"],
                                    "entry_price": entry_price,
                                    "exit_price": exit_price,
                                    "qty": match_qty,
                                    "pnl": round(pnl, 2),
                                    "strategy": head.get("strategy", "unknown"),
                                    "exit_strategy": fill_strategy,  # Track what closed it
                                }
                            )

                            head["qty"] -= match_qty
                            remaining_qty -= match_qty

                            if head["qty"] <= 0.0000001:
                                position_stack.popleft()

            # Collect remaining open positions for this symbol
            while position_stack:
                pos = position_stack.popleft()
                all_open_positions.append(
                    {
                        "symbol": symbol,
                        "side": pos["side"],
                        "qty": pos["qty"],
                        "entry_price": pos["price"],
                        "entry_time": pos["time"],
                        "strategy": pos.get("strategy", "unknown"),
                    }
                )

        return trades, all_open_positions
