from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.core.domain import Bar, OrderIntent, OrderSide, OrderType, Side
from src.core.logger import StructuredLogger
from src.engine.execution import ExecutionEngine


def _ensure_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    return datetime.now(timezone.utc)


@dataclass
class _PendingOrder:
    id: str
    intent: OrderIntent
    status: str
    submitted_at: datetime


class BacktestOrderExecutor:
    """
    Deterministic OrderExecutor for portfolio backtests.

    Key behaviors:
    - Orders are placed via `submit(intent)` by the ExecutionEngine.
    - BacktestRunner drives fills by calling:
      - `fill_pending_for_bar(...)` (limit/market fills using bar OHLC)
      - `maybe_trigger_bracket_exit(...)` (broker-managed stop/target using bar extremes)
    - All fills are routed through `engine.on_fill(...)` so the engine's normal
      PositionManager / RiskManager logic applies.
    """

    broker_managed_exits: bool = True

    def __init__(self, logger: StructuredLogger, *, initial_cash: float = 100000.0):
        self.logger = logger
        self._initial_cash: float = float(initial_cash)
        self.cash: float = float(initial_cash)
        self._positions: Dict[str, Dict[str, float]] = {}  # symbol -> {qty, avg_price}
        self._orders: list[_PendingOrder] = []
        # Performance: Symbol-indexed pending orders for O(1) lookup
        self._pending_by_symbol: Dict[str, list[_PendingOrder]] = {}
        # Public, analysis-friendly records (stable keys).
        self.fills: list[Dict[str, Any]] = []

        self._risk_cfg: Dict[str, Any] = {}
        self._backtest_cfg: Dict[str, Any] = {}
        self._max_open_order_age_sec: int = 0
        # P1: Bracket exit mode ("stop_first" or "best_exit")
        self._bracket_exit_mode: str = "stop_first"
        # Partial fill configuration
        self._partial_fill_mode: str = "none"  # none|fixed|volume_aware
        self._partial_fill_pct: float = 1.0  # for fixed mode
        self._partial_fill_rate: float = (
            0.1  # for volume_aware: capture 10% of bar volume
        )
        # Slippage configuration
        self._slippage_mode: str = "fixed"  # fixed|volume_impact
        self._slippage_impact_mult: float = 5.0  # multiplier for volume impact
        # Spread configuration
        self._spread_mode: str = "fixed"  # fixed|atr_based
        # Daily equity reset for regime analysis backtests
        self._daily_equity_reset: bool = False

    def set_risk_config(self, risk_cfg: Optional[Dict[str, Any]]) -> None:
        self._risk_cfg = dict(risk_cfg) if isinstance(risk_cfg, dict) else {}
        # Bracket exit mode from risk config
        self._bracket_exit_mode = str(
            self._risk_cfg.get("bracket_exit_mode", "stop_first")
        ).lower()

    def set_backtest_config(self, backtest_cfg: Optional[Dict[str, Any]]) -> None:
        """Configure backtest-specific realism settings."""
        self._backtest_cfg = (
            dict(backtest_cfg) if isinstance(backtest_cfg, dict) else {}
        )

        # Partial fill mode: none|fixed|volume_aware
        self._partial_fill_mode = str(
            self._backtest_cfg.get("partial_fill_mode", "none")
        ).lower()

        # Fixed partial fill percentage (for mode=fixed)
        try:
            self._partial_fill_pct = float(
                self._backtest_cfg.get("partial_fill_pct", 1.0)
            )
            self._partial_fill_pct = max(0.1, min(1.0, self._partial_fill_pct))
        except (TypeError, ValueError):
            self._partial_fill_pct = 1.0

        # Volume-aware fill rate (for mode=volume_aware)
        try:
            self._partial_fill_rate = float(
                self._backtest_cfg.get("partial_fill_rate", 0.1)
            )
            self._partial_fill_rate = max(0.01, min(1.0, self._partial_fill_rate))
        except (TypeError, ValueError):
            self._partial_fill_rate = 0.1

        # Slippage mode: fixed|volume_impact
        self._slippage_mode = str(
            self._backtest_cfg.get("slippage_mode", "fixed")
        ).lower()

        # Volume impact multiplier (for slippage_mode=volume_impact)
        try:
            self._slippage_impact_mult = float(
                self._backtest_cfg.get("slippage_impact_mult", 5.0)
            )
        except (TypeError, ValueError):
            self._slippage_impact_mult = 5.0

        # Spread mode: fixed|atr_based
        self._spread_mode = str(self._backtest_cfg.get("spread_mode", "fixed")).lower()

        # Daily equity reset: start each day fresh with initial_cash
        self._daily_equity_reset = bool(
            self._backtest_cfg.get("daily_equity_reset", False)
        )

        # Check if advanced exits are enabled (from risk config)
        # If so, disable broker_managed_exits to let PositionManager handle exits
        adv_exits = self._risk_cfg.get("advanced_exits", {})
        if isinstance(adv_exits, dict) and adv_exits.get("enabled", False):
            self.broker_managed_exits = False
            self.logger.info(
                "Advanced exits enabled, broker_managed_exits disabled",
                trailing_stop=adv_exits.get("trailing_stop", {}).get("enabled", False),
                partial_exits=adv_exits.get("partial_exits", {}).get("enabled", False),
                regime_aware_stops=adv_exits.get("regime_aware_stops", True),
            )

    def set_max_open_order_age_sec(self, value: Any) -> None:
        try:
            self._max_open_order_age_sec = int(value or 0)
        except Exception:
            self._max_open_order_age_sec = 0

    def reset_daily_equity(self) -> None:
        """Reset cash to initial value for regime analysis backtests."""
        if self._daily_equity_reset:
            old_cash = self.cash
            self.cash = self._initial_cash
            self.logger.info(
                "Daily equity reset",
                old_cash=old_cash,
                new_cash=self.cash,
            )

    def submit(self, intent: OrderIntent) -> Dict[str, Any]:
        submitted_at = _ensure_dt((intent.meta or {}).get("created_at"))
        order_id = f"bt-{len(self._orders) + 1}"
        order = _PendingOrder(
            id=order_id,
            intent=intent,
            status="new",
            submitted_at=submitted_at,
        )
        self._orders.append(order)
        # Performance: Index by symbol for O(1) lookup in fill_pending_for_bar
        sym = str(intent.symbol)
        if sym not in self._pending_by_symbol:
            self._pending_by_symbol[sym] = []
        self._pending_by_symbol[sym].append(order)
        self.logger.info(
            "Backtest order submitted",
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side.value,
            qty=float(intent.qty),
            order_type=intent.order_type.value,
            limit_price=(
                float(intent.limit_price) if intent.limit_price is not None else None
            ),
            correlation_id=intent.correlation_id,
            strategy=intent.strategy,
        )
        return {"id": order_id, "status": "new"}

    def _slippage_bps(self) -> float:
        try:
            return float(self._risk_cfg.get("slippage_bps", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _spread_bps(self) -> float:
        try:
            return float(self._risk_cfg.get("spread_bps", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _commission_for(self, qty: float) -> float:
        cps = float(self._risk_cfg.get("commission_per_share", 0.0) or 0.0)
        min_c = float(self._risk_cfg.get("min_commission", 0.0) or 0.0)
        if cps <= 0.0 and min_c <= 0.0:
            return 0.0
        return float(max(min_c, cps * float(qty)))

    def _apply_spread(self, side: str, price: float, *, atr_pct: float = 0.0) -> float:
        """Apply bid-ask spread to a fill price.

        Supports two modes:
        - fixed: standard BPS spread
        - atr_based: spread scales with ATR volatility
        """
        bps = self._spread_bps()
        if bps <= 0.0:
            return float(price)

        # Calculate effective spread based on mode
        if self._spread_mode == "atr_based" and atr_pct > 0:
            # ATR-based: higher volatility = wider spreads
            # Assuming avg ATR is ~1%, scale spread proportionally
            avg_atr_pct = 0.01  # 1% baseline
            volatility_mult = atr_pct / avg_atr_pct
            effective_bps = bps * max(0.5, min(3.0, volatility_mult))
        else:
            effective_bps = bps

        half = (effective_bps / 10000.0) / 2.0
        if str(side).lower() == "buy":
            return float(price) * (1.0 + half)
        return float(price) * (1.0 - half)

    def _apply_slippage(
        self,
        side: str,
        price: float,
        *,
        order_qty: float = 0.0,
        bar_volume: float = 0.0,
    ) -> float:
        """Apply slippage to a fill price.

        Supports two modes:
        - fixed: standard BPS slippage
        - volume_impact: slippage increases with order size relative to bar volume
        """
        bps = self._slippage_bps()
        if bps <= 0.0:
            return float(price)

        # Calculate effective slippage based on mode
        if self._slippage_mode == "volume_impact" and bar_volume > 0 and order_qty > 0:
            # Volume impact: larger orders relative to volume get more slippage
            volume_ratio = float(order_qty) / float(bar_volume)
            effective_bps = bps * (1.0 + volume_ratio * self._slippage_impact_mult)
        else:
            effective_bps = bps

        mult = 1.0 + (effective_bps / 10000.0)
        # Buys pay more; sells receive less.
        if str(side).lower() == "buy":
            return float(price) * mult
        return float(price) / mult

    def _calculate_fill_qty(self, order_qty: float, bar_volume: float) -> float:
        """Calculate actual fill quantity based on partial fill mode.

        Supports three modes:
        - none: full fills (100%)
        - fixed: fill at configured percentage
        - volume_aware: fill based on order size vs bar volume
        """
        if self._partial_fill_mode == "none":
            return float(order_qty)

        if self._partial_fill_mode == "fixed":
            return float(order_qty) * self._partial_fill_pct

        if self._partial_fill_mode == "volume_aware" and bar_volume > 0:
            # Can only capture a fraction of bar volume
            max_capturable = float(bar_volume) * self._partial_fill_rate
            return min(float(order_qty), max_capturable)

        return float(order_qty)

    def _record_fill(
        self,
        *,
        order_id: str,
        intent: OrderIntent,
        fill_qty: float,
        fill_price: float,
        filled_at: datetime,
        kind: str,
    ) -> None:
        self.fills.append(
            {
                "id": order_id,
                "symbol": intent.symbol,
                "side": intent.side.value,
                "qty": float(fill_qty),
                "type": intent.order_type.value,
                "strategy": intent.strategy,
                "correlation_id": intent.correlation_id,
                "status": "filled",
                "fill_price": float(fill_price),
                "filled_at": filled_at,
                "kind": kind,
            }
        )

    def _update_cash_and_positions(
        self, *, symbol: str, side: str, qty: float, price: float
    ) -> None:
        qty_f = float(qty)
        px = float(price)
        commission = self._commission_for(qty_f)

        if str(side).lower() == "buy":
            self.cash -= (qty_f * px) + commission
        else:
            self.cash += (qty_f * px) - commission

        pos = self._positions.setdefault(symbol, {"qty": 0.0, "avg_price": 0.0})
        prev_qty = float(pos["qty"])

        if str(side).lower() == "buy":
            new_qty = prev_qty + qty_f
        else:
            new_qty = prev_qty - qty_f

        # Simple average price handling: only maintain avg on same-direction adds.
        # If closing (reducing size) or flipping, avg_price doesn't mathematically change
        # for the remaining portion until the flip occurs, but here we simplify:
        # If we flip from short to long or long to short, or start from 0, reset avg_price.
        is_same_dir_increase = (prev_qty > 0 and new_qty > prev_qty) or (
            prev_qty < 0 and new_qty < prev_qty
        )

        if math.isclose(prev_qty, 0.0, abs_tol=1e-9):
            pos["avg_price"] = px
        elif is_same_dir_increase:
            total_cost = (abs(prev_qty) * float(pos["avg_price"])) + (qty_f * px)
            pos["avg_price"] = total_cost / max(1e-9, abs(new_qty))

        # If we crossed zero or reached zero, reset logic might apply, but basic avg price
        # maintenance for valid open positions is covered above.

        if math.isclose(new_qty, 0.0, abs_tol=1e-9):
            pos["qty"] = 0.0
            pos["avg_price"] = 0.0
        else:
            pos["qty"] = new_qty

    def _maybe_fill_price_for_order(
        self, intent: OrderIntent, bar: Bar
    ) -> Optional[float]:
        """
        Deterministic fill model:
        - MARKET: fill at bar.open
        - LIMIT:
          - BUY: fill if low <= limit; price = min(open, limit)
          - SELL: fill if high >= limit; price = max(open, limit)
        """
        if intent.order_type == OrderType.MARKET:
            return float(bar.open)

        if intent.order_type != OrderType.LIMIT:
            return None

        limit_price = intent.limit_price
        if limit_price is None:
            return None
        limit_f = float(limit_price)

        if intent.side == OrderSide.BUY:
            if float(bar.low) <= limit_f:
                return float(min(float(bar.open), limit_f))
            return None

        # SELL
        if float(bar.high) >= limit_f:
            return float(max(float(bar.open), limit_f))
        return None

    def fill_pending_for_bar(
        self, engine: ExecutionEngine, symbol: str, bar: Bar
    ) -> None:
        """
        Attempt fills for pending orders for `symbol` using this bar's OHLC.
        Fills only occur on bars strictly after the order was submitted.

        Performance: Uses symbol-indexed orders for O(1) lookup instead of O(n) scan.
        """
        now = _ensure_dt(bar.time)

        # P3: Check and cancel expired orders at the start of each bar
        self._cancel_all_expired_orders(now)

        # Performance: O(1) symbol lookup instead of iterating all orders
        pending = self._pending_by_symbol.get(symbol, [])
        for o in pending:
            if not self._can_fill_order(o, symbol, now):
                continue

            if self._check_and_cancel_expired(o, now):
                continue

            raw_px = self._maybe_fill_price_for_order(o.intent, bar)
            if raw_px is None or raw_px <= 0.0:
                continue

            self._execute_order_fill(engine, o, raw_px, now, symbol, bar)

    def _can_fill_order(self, o: _PendingOrder, symbol: str, now: datetime) -> bool:
        if o.status != "new":
            return False
        if o.intent.symbol != symbol:
            return False
        if now <= o.submitted_at:
            return False
        return True

    def _check_and_cancel_expired(self, o: _PendingOrder, now: datetime) -> bool:
        if self._max_open_order_age_sec <= 0:
            return False

        try:
            age = (now - o.submitted_at).total_seconds()
        except Exception:
            age = 0.0

        if age >= float(self._max_open_order_age_sec):
            o.status = "canceled"
            self.logger.info(
                "Backtest order canceled due to age",
                order_id=o.id,
                symbol=o.intent.symbol,
                correlation_id=o.intent.correlation_id,
                age_sec=float(age),
                max_age_sec=int(self._max_open_order_age_sec),
            )
            return True
        return False

    def _cancel_all_expired_orders(self, now: datetime) -> None:
        """P3: Check and cancel all expired orders at start of each bar."""
        if self._max_open_order_age_sec <= 0:
            return
        for o in self._orders:
            if o.status == "new":
                self._check_and_cancel_expired(o, now)

    def _execute_order_fill(
        self,
        engine: ExecutionEngine,
        o: _PendingOrder,
        raw_px: float,
        fill_ts: datetime,
        symbol: str,
        bar: Bar,
    ) -> None:
        order_qty = float(o.intent.qty)
        bar_volume = float(bar.volume) if bar.volume > 0 else 0.0

        # Calculate fill quantity using volume-aware logic
        fill_qty = self._calculate_fill_qty(order_qty, bar_volume)
        if fill_qty <= 0:
            return

        # Apply spread first, then volume-aware slippage
        spread_px = self._apply_spread(o.intent.side.value, raw_px)
        fill_px = self._apply_slippage(
            o.intent.side.value,
            spread_px,
            order_qty=fill_qty,
            bar_volume=bar_volume,
        )

        # Update portfolio model first (cash), then engine state.
        self._update_cash_and_positions(
            symbol=symbol,
            side=o.intent.side.value,
            qty=fill_qty,
            price=fill_px,
        )
        engine.on_fill(
            {
                "symbol": symbol,
                "side": o.intent.side.value,
                "qty": fill_qty,
                "price": float(fill_px),
                "timestamp": fill_ts,
                "strategy": o.intent.strategy,
                "correlation_id": o.intent.correlation_id,
                "broker_order_id": o.id,
                "trade_event": "fill",
            }
        )

        o.status = "filled"
        self._record_fill(
            order_id=o.id,
            intent=o.intent,
            fill_qty=fill_qty,
            fill_price=fill_px,
            filled_at=fill_ts,
            kind="order",
        )
        self.logger.info(
            "Backtest order filled",
            order_id=o.id,
            symbol=symbol,
            side=o.intent.side.value,
            price=float(fill_px),
            qty=float(fill_qty),
            correlation_id=o.intent.correlation_id,
        )

    def maybe_trigger_bracket_exit(
        self, engine: ExecutionEngine, symbol: str, bar: Bar
    ) -> None:
        """
        Simulate broker-managed stop/target exits using intrabar extremes.
        """
        state = engine.symbol_states.get(symbol)
        pos = state.position if state is not None else None
        if pos is None:
            return

        exit_price, reason = self._calculate_bracket_exit(pos, bar)
        if exit_price is None:
            return

        exit_side = OrderSide.SELL if pos.side == Side.LONG else OrderSide.BUY
        now = _ensure_dt(bar.time)

        exit_px = self._apply_slippage(
            exit_side.value,
            self._apply_spread(exit_side.value, exit_price),
        )
        qty = float(pos.qty)
        if qty <= 0.0 or exit_px <= 0.0:
            return

        # Portfolio model first.
        self._update_cash_and_positions(
            symbol=symbol, side=exit_side.value, qty=qty, price=exit_px
        )

        # Engine fill to update position/risk/trades.
        engine.on_fill(
            {
                "symbol": symbol,
                "side": exit_side.value,
                "qty": qty,
                "price": float(exit_px),
                "timestamp": now,
                "strategy": str(getattr(pos, "strategy", "") or "unknown"),
                "correlation_id": str(getattr(pos, "correlation_id", "") or ""),
                "broker_order_id": f"bt-bracket-{symbol}-{now.isoformat()}",
                "trade_event": "fill",
            }
        )

        # Record fill in analyzer format.
        intent = OrderIntent(
            symbol=symbol,
            side=exit_side,
            qty=qty,
            order_type=OrderType.MARKET,
            limit_price=None,
            time_in_force="day",
            correlation_id=str(getattr(pos, "correlation_id", "") or ""),
            strategy=str(getattr(pos, "strategy", "") or "unknown"),
            stop_loss=None,
            take_profit=None,
            meta={"created_at": now.isoformat()},
        )
        self._record_fill(
            order_id=f"bt-bracket-{len(self.fills) + 1}",
            intent=intent,
            fill_qty=qty,
            fill_price=exit_px,
            filled_at=now,
            kind="bracket",
        )
        self.logger.info(
            "Backtest bracket exit filled",
            symbol=symbol,
            side=exit_side.value,
            price=float(exit_px),
            qty=float(qty),
            reason=reason,
        )

    def _check_bracket_hits(
        self,
        pos: Any,
        bar: Bar,
        stop_price: Optional[float],
        target_price: Optional[float],
    ) -> tuple[bool, bool]:
        """Check if stop or target were hit during the bar."""
        low = float(bar.low)
        high = float(bar.high)
        hit_stop = False
        hit_target = False

        if pos.side == Side.LONG:
            if stop_price is not None and low <= float(stop_price):
                hit_stop = True
            if target_price is not None and high >= float(target_price):
                hit_target = True
        else:
            if stop_price is not None and high >= float(stop_price):
                hit_stop = True
            if target_price is not None and low <= float(target_price):
                hit_target = True

        return hit_stop, hit_target

    def _calculate_gap_aware_exit(
        self, pos: Any, open_px: float, price: float, is_stop: bool
    ) -> float:
        """Calculate exit price accounting for gaps past stop/target."""
        is_long = pos.side == Side.LONG

        # For stops: gap through = worse fill (fill at open)
        # For targets: gap through = better fill (fill at open)
        if is_stop:
            if is_long:
                return min(open_px, price)
            else:
                return max(open_px, price)
        else:
            if is_long:
                return max(open_px, price)
            else:
                return min(open_px, price)

    def _calculate_bracket_exit(
        self, pos: Any, bar: Bar
    ) -> tuple[Optional[float], str]:
        stop_price = getattr(pos, "stop_price", None)
        target_price = getattr(pos, "target_price", None)
        if stop_price is None and target_price is None:
            return None, ""

        hit_stop, hit_target = self._check_bracket_hits(
            pos, bar, stop_price, target_price
        )
        if not (hit_stop or hit_target):
            return None, ""

        open_px = float(bar.open)

        # P1: Choose exit based on mode when both are hit
        if hit_stop and hit_target:
            if self._bracket_exit_mode == "best_exit":
                # Best exit mode: target wins (more favorable outcome)
                use_stop = False
            else:
                # Default stop_first mode: stop wins (conservative)
                use_stop = True
        elif hit_stop:
            use_stop = True
        else:
            use_stop = False

        if use_stop:
            assert stop_price is not None
            exit_price = self._calculate_gap_aware_exit(
                pos,
                open_px,
                float(stop_price),
                is_stop=True,
            )
            return exit_price, "STOP_HIT"
        else:
            assert target_price is not None
            exit_price = self._calculate_gap_aware_exit(
                pos,
                open_px,
                float(target_price),
                is_stop=False,
            )
            return exit_price, "TARGET_HIT"

    def close_all_positions(
        self,
        engine: ExecutionEngine,
        *,
        timestamp: datetime,
        prices: Dict[str, float],
        reason: str,
    ) -> None:
        ts = _ensure_dt(timestamp)
        for sym, st in engine.symbol_states.items():
            pos = getattr(st, "position", None)
            if pos is None:
                continue
            mark = float(prices.get(sym, 0.0) or 0.0)
            if mark <= 0.0:
                continue

            exit_side = OrderSide.SELL if pos.side == Side.LONG else OrderSide.BUY
            qty = float(pos.qty)
            px = self._apply_slippage(
                exit_side.value, self._apply_spread(exit_side.value, mark)
            )

            self._update_cash_and_positions(
                symbol=sym, side=exit_side.value, qty=qty, price=px
            )
            engine.on_fill(
                {
                    "symbol": sym,
                    "side": exit_side.value,
                    "qty": qty,
                    "price": float(px),
                    "timestamp": ts,
                    "strategy": str(getattr(pos, "strategy", "") or "unknown"),
                    "correlation_id": str(getattr(pos, "correlation_id", "") or ""),
                    "broker_order_id": f"bt-eod-{sym}-{ts.isoformat()}",
                    "trade_event": "fill",
                }
            )

            intent = OrderIntent(
                symbol=sym,
                side=exit_side,
                qty=qty,
                order_type=OrderType.MARKET,
                limit_price=None,
                time_in_force="day",
                correlation_id=str(getattr(pos, "correlation_id", "") or ""),
                strategy=str(getattr(pos, "strategy", "") or "unknown"),
                stop_loss=None,
                take_profit=None,
                meta={"created_at": ts.isoformat(), "exit_reason": reason},
            )
            self._record_fill(
                order_id=f"bt-eod-{len(self.fills) + 1}",
                intent=intent,
                fill_qty=qty,
                fill_price=px,
                filled_at=ts,
                kind="eod",
            )

    def cancel_all_orders(self) -> None:
        for o in self._orders:
            if o.status == "new":
                o.status = "canceled"
        # Performance: Clear symbol index (orders still in _orders for history)
        self._pending_by_symbol.clear()


# Backwards-compatible name used in tests/runner wiring.
MockOrderExecutor = BacktestOrderExecutor
