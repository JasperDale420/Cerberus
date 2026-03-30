from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

# L1 fix: Named constant for floating-point quantity comparisons
# Prevents infinite loops and handles very small position sizes
QTY_EPSILON = 1e-7


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
        Matches trades FIFO and calculates realized + unrealized PnL.
        """
        fills = fills or []
        trades, open_positions = self._match_trades(fills)

        # 1. Realized Metrics
        realized = self._calculate_realized_metrics(trades)
        total_closed_pnl = realized["total_closed_pnl"]

        # 2. Unrealized Metrics
        unrealized = self._calculate_unrealized_metrics(open_positions, current_prices)
        open_pnl = unrealized["open_pnl"]

        # 3. Totals
        total_equity_pnl = total_closed_pnl + open_pnl
        total_equity = self.initial_cash + total_equity_pnl
        return_pct = (total_equity_pnl / self.initial_cash) * 100 if self.initial_cash > 0 else 0.0

        # 4. Drawdown
        max_drawdown = self._calculate_drawdown(trades, open_pnl, total_closed_pnl)

        return {
            "total_trades": realized["total_trades"],
            "total_closed_pnl": round(total_closed_pnl, 2),
            "open_pnl": round(open_pnl, 2),
            "total_pnl": round(total_equity_pnl, 2),
            "total_equity": round(total_equity, 2),
            "win_rate": round(realized["win_rate"], 4),
            "profit_factor": round(realized["profit_factor"], 4),
            "average_pnl": round(realized["average_pnl"], 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "return_pct": round(return_pct, 2),
            "trades": realized["sorted_trades"],
            "open_positions": unrealized["valued_open_positions"],
        }

    def _calculate_realized_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trades:
            return {
                "total_trades": 0,
                "total_closed_pnl": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_pnl": 0.0,
                "sorted_trades": [],
            }

        pnls = [t["pnl"] for t in trades]
        total_closed_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_trades = len(trades)

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        avg_pnl = total_closed_pnl / total_trades if total_trades > 0 else 0.0

        return {
            "total_trades": total_trades,
            "total_closed_pnl": total_closed_pnl,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_pnl": avg_pnl,
            "sorted_trades": sorted(trades, key=lambda x: x["exit_time"]),
        }

    def _calculate_unrealized_metrics(
        self,
        open_positions: List[Dict[str, Any]],
        current_prices: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        if not current_prices:
            return {"open_pnl": 0.0, "valued_open_positions": open_positions}

        open_pnl = 0.0
        valued = []
        for pos in open_positions:
            symbol = pos["symbol"]
            current_price = current_prices.get(symbol)
            unrealized = 0.0

            if current_price:
                if pos["side"] == "buy":
                    unrealized = (current_price - pos["entry_price"]) * pos["qty"]
                else:
                    unrealized = (pos["entry_price"] - current_price) * pos["qty"]
                open_pnl += unrealized

            # Copy and update
            p_copy = pos.copy()
            p_copy["current_price"] = current_price
            p_copy["unrealized_pnl"] = round(unrealized, 2)
            valued.append(p_copy)

        return {"open_pnl": open_pnl, "valued_open_positions": valued}

    def _calculate_drawdown(self, trades: List[Dict[str, Any]], open_pnl: float, closed_pnl: float) -> float:
        """
        Calculate maximum drawdown based on closed trades.

        H3 fix: This method calculates drawdown from the closed-trade equity curve.
        The final check includes open_pnl for current state comparison, but peak
        tracking is based only on closed trades to maintain consistency.

        Returns:
            Maximum drawdown as a decimal (e.g., 0.10 = 10% drawdown)
        """
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

        # H3 fix: Calculate current drawdown from closed-trade peak, including open PnL
        # This shows current drawdown state without updating peak (open positions may reverse)
        current_equity = self.initial_cash + closed_pnl + open_pnl
        current_drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
        # Only count as drawdown if current equity is below peak (not a new high)
        if current_drawdown > 0:
            max_drawdown = max(max_drawdown, current_drawdown)
        return max_drawdown

    def _match_trades(self, fills: List[Dict[str, Any]]) -> Any:
        # Returning Tuple[List[Dict], List[Dict]] - defined as Any to avoid extensive typing imports inline

        # Group fills by symbol
        fills_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for fill in fills:
            sym = fill["symbol"]
            fills_by_symbol.setdefault(sym, []).append(fill)

        trades = []
        all_open_positions = []

        for symbol, sym_fills in fills_by_symbol.items():
            t, p = self._process_symbol_fills(symbol, sym_fills)
            trades.extend(t)
            all_open_positions.extend(p)

        return trades, all_open_positions

    def _process_symbol_fills(self, symbol: str, sym_fills: List[Dict[str, Any]]) -> Any:
        sym_fills.sort(key=lambda x: x["filled_at"])
        position_stack: deque[Dict[str, Any]] = deque()
        trades: List[Dict[str, Any]] = []

        for fill in sym_fills:
            self._apply_fill_to_stack(symbol, fill, position_stack, trades)

        # Collect remaining open positions for this symbol
        open_positions = []
        while position_stack:
            pos = position_stack.popleft()
            open_positions.append(
                {
                    "symbol": symbol,
                    "side": pos["side"],
                    "qty": pos["qty"],
                    "entry_price": pos["price"],
                    "entry_time": pos["time"],
                    "strategy": pos.get("strategy", "unknown"),
                }
            )
        return trades, open_positions

    def _apply_fill_to_stack(
        self,
        symbol: str,
        fill: Dict[str, Any],
        stack: deque[Dict[str, Any]],
        trades: List[Dict[str, Any]],
    ) -> None:
        qty = float(fill["qty"])
        price = float(fill["fill_price"])
        side = fill["side"]
        fill_strategy = fill.get("strategy", "unknown")
        fill_regime = fill.get("regime_labels", {})
        remaining_qty = qty

        while remaining_qty > QTY_EPSILON:
            if not stack:
                self._open_new_stack_position(
                    stack,
                    side,
                    remaining_qty,
                    price,
                    fill["filled_at"],
                    fill_strategy,
                    fill_regime,
                )
                remaining_qty = 0
            else:
                remaining_qty = self._process_stack_item(
                    symbol,
                    item=stack[0],
                    side=side,
                    price=price,
                    qty=remaining_qty,
                    fill_strategy=fill_strategy,
                    filled_at=fill["filled_at"],
                    stack=stack,
                    trades=trades,
                    fill_regime=fill_regime,
                )

    def _open_new_stack_position(
        self,
        stack: deque[Dict[str, Any]],
        side: str,
        qty: float,
        price: float,
        filled_at: datetime,
        strategy: str,
        regime_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        stack.append(
            {
                "side": side,
                "qty": qty,
                "price": price,
                "time": filled_at,
                "strategy": strategy,
                "regime_labels": regime_labels or {},
            }
        )

    def _process_stack_item(
        self,
        symbol: str,
        item: Dict[str, Any],
        side: str,
        price: float,
        qty: float,
        fill_strategy: str,
        filled_at: datetime,
        stack: deque[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        fill_regime: Optional[Dict[str, str]] = None,
    ) -> float:
        if item["side"] == side:
            # Add to position
            stack.append(
                {
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "time": filled_at,
                    "strategy": fill_strategy,
                    "regime_labels": fill_regime or {},
                }
            )
            return 0.0
        else:
            # Close/Reduce position
            head_qty = float(item["qty"])
            match_qty = min(qty, head_qty)

            entry_price = float(item["price"])
            pnl = (price - entry_price) * match_qty if item["side"] == "buy" else (entry_price - price) * match_qty

            entry_regime = item.get("regime_labels", {})
            exit_regime = fill_regime or {}

            trades.append(
                {
                    "symbol": symbol,
                    "side": item["side"],
                    "entry_time": item["time"],
                    "exit_time": filled_at,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "qty": match_qty,
                    "pnl": round(pnl, 2),
                    "strategy": item.get("strategy", "unknown"),
                    "exit_strategy": fill_strategy,
                    "entry_regime_trend": entry_regime.get("trend", "unknown"),
                    "entry_regime_vol": entry_regime.get("vol", "unknown"),
                    "exit_regime_trend": exit_regime.get("trend", "unknown"),
                    "exit_regime_vol": exit_regime.get("vol", "unknown"),
                }
            )

            item["qty"] = head_qty - match_qty
            if item["qty"] <= QTY_EPSILON:
                stack.popleft()

            return qty - match_qty
