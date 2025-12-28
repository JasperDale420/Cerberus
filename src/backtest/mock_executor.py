from __future__ import annotations

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
        self.cash: float = float(initial_cash)
        self._positions: Dict[str, Dict[str, float]] = {}  # symbol -> {qty, avg_price}
        self._orders: list[_PendingOrder] = []
        # Public, analysis-friendly records (stable keys).
        self.fills: list[Dict[str, Any]] = []

        self._risk_cfg: Dict[str, Any] = {}
        self._max_open_order_age_sec: int = 0

    def set_risk_config(self, risk_cfg: Optional[Dict[str, Any]]) -> None:
        self._risk_cfg = dict(risk_cfg) if isinstance(risk_cfg, dict) else {}

    def set_max_open_order_age_sec(self, value: Any) -> None:
        try:
            self._max_open_order_age_sec = int(value or 0)
        except Exception:
            self._max_open_order_age_sec = 0

    def submit(self, intent: OrderIntent) -> Dict[str, Any]:
        submitted_at = _ensure_dt((intent.meta or {}).get("created_at"))
        order_id = f"bt-{len(self._orders) + 1}"
        self._orders.append(
            _PendingOrder(
                id=order_id,
                intent=intent,
                status="new",
                submitted_at=submitted_at,
            )
        )
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

    def _apply_spread(self, side: str, price: float) -> float:
        bps = self._spread_bps()
        if bps <= 0.0:
            return float(price)
        half = (bps / 10000.0) / 2.0
        if str(side).lower() == "buy":
            return float(price) * (1.0 + half)
        return float(price) * (1.0 - half)

    def _apply_slippage(self, side: str, price: float) -> float:
        bps = self._slippage_bps()
        if bps <= 0.0:
            return float(price)
        mult = 1.0 + (bps / 10000.0)
        # Buys pay more; sells receive less.
        if str(side).lower() == "buy":
            return float(price) * mult
        return float(price) / mult

    def _record_fill(
        self,
        *,
        order_id: str,
        intent: OrderIntent,
        fill_price: float,
        filled_at: datetime,
        kind: str,
    ) -> None:
        self.fills.append(
            {
                "id": order_id,
                "symbol": intent.symbol,
                "side": intent.side.value,
                "qty": float(intent.qty),
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

        pos = self._positions.get(symbol)
        if pos is None:
            pos = {"qty": 0.0, "avg_price": 0.0}
            self._positions[symbol] = pos

        prev_qty = float(pos["qty"])
        if str(side).lower() == "buy":
            new_qty = prev_qty + qty_f
        else:
            new_qty = prev_qty - qty_f

        # Simple average price handling: only maintain avg on same-direction adds.
        if (
            prev_qty == 0.0
            or (prev_qty > 0 and new_qty > 0)
            or (prev_qty < 0 and new_qty < 0)
        ):
            if prev_qty == 0.0:
                pos["avg_price"] = px
            else:
                total_cost = (abs(prev_qty) * float(pos["avg_price"])) + (qty_f * px)
                pos["avg_price"] = total_cost / max(1e-9, abs(new_qty))
        if new_qty == 0.0:
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
        """
        now = _ensure_dt(bar.time)
        for o in self._orders:
            if o.status != "new":
                continue
            if o.intent.symbol != symbol:
                continue
            if now <= o.submitted_at:
                continue

            if self._max_open_order_age_sec > 0:
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
                    continue

            raw_px = self._maybe_fill_price_for_order(o.intent, bar)
            if raw_px is None or raw_px <= 0.0:
                continue

            fill_px = self._apply_slippage(
                o.intent.side.value,
                self._apply_spread(o.intent.side.value, raw_px),
            )
            fill_ts = now

            # Update portfolio model first (cash), then engine state.
            self._update_cash_and_positions(
                symbol=symbol,
                side=o.intent.side.value,
                qty=float(o.intent.qty),
                price=fill_px,
            )
            engine.on_fill(
                {
                    "symbol": symbol,
                    "side": o.intent.side.value,
                    "qty": float(o.intent.qty),
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
                qty=float(o.intent.qty),
                correlation_id=o.intent.correlation_id,
            )

    def maybe_trigger_bracket_exit(
        self, engine: ExecutionEngine, symbol: str, bar: Bar
    ) -> None:
        """
        Simulate broker-managed stop/target exits using intrabar extremes.
        Stop has priority if both stop and target cross within the same bar.
        """
        state = engine.symbol_states.get(symbol)
        pos = state.position if state is not None else None
        if pos is None:
            return

        stop_price = getattr(pos, "stop_price", None)
        target_price = getattr(pos, "target_price", None)
        if stop_price is None and target_price is None:
            return

        low = float(bar.low)
        high = float(bar.high)
        open_px = float(bar.open)

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

        if not (hit_stop or hit_target):
            return

        # Gap-aware fill: if the open is past the stop/target, fill at open (worse).
        if hit_stop:
            stop_px = float(stop_price)  # type: ignore[arg-type]
            if pos.side == Side.LONG:
                exit_price = open_px if open_px <= stop_px else stop_px
            else:
                exit_price = open_px if open_px >= stop_px else stop_px
        else:
            tgt_px = float(target_price)  # type: ignore[arg-type]
            if pos.side == Side.LONG:
                exit_price = open_px if open_px >= tgt_px else tgt_px
            else:
                exit_price = open_px if open_px <= tgt_px else tgt_px

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
            reason="STOP_HIT" if hit_stop else "TARGET_HIT",
        )

    def close_all_positions(
        self,
        engine: ExecutionEngine,
        *,
        timestamp: datetime,
        prices: Dict[str, float],
        reason: str,
    ) -> None:
        ts = _ensure_dt(timestamp)
        for sym, st in list(engine.symbol_states.items()):
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
                fill_price=px,
                filled_at=ts,
                kind="eod",
            )

    def cancel_all_orders(self) -> None:
        for o in self._orders:
            if o.status == "new":
                o.status = "canceled"


# Backwards-compatible name used in tests/runner wiring.
MockOrderExecutor = BacktestOrderExecutor
