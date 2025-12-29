from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.core.domain import (
    MarketState,
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
    Regime,
    Side,
    SymbolState,
)
from src.core.type_utils import safe_float, safe_int

# M1 fix: Module-level logger for debug visibility without breaking trading
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExitDecision:
    intent: Optional[OrderIntent]
    reason: Optional[str]


@dataclass(frozen=True)
class ClosedTradeInfo:
    symbol: str
    strategy: str
    regime_at_entry: str
    regime_at_exit: str
    side: str
    qty: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl_gross: float
    pnl_net: float
    initial_risk: Optional[float]
    mae_r: float
    mfe_r: float
    commission: float
    slippage_estimate: float
    pnl_r: Optional[float]
    holding_period_seconds: Optional[float]
    features_json: Optional[dict]
    correlation_id: str


@dataclass(frozen=True)
class FillDecision:
    event: str  # opened | increased | reduced | closed | ignored
    realized_pnl_delta: float
    close_qty: float
    closed_trade: Optional[ClosedTradeInfo]


class PositionManager:
    """
    Best-effort, deterministic exit logic when stop/target are not broker-managed.
    """

    def _validate_fill(self, fill: Dict[str, Any]) -> Optional[str]:
        """
        Validate fill event data for safety and correctness.

        Performs comprehensive validation of fill data to prevent malformed
        events from corrupting position state. Checks for positive, finite,
        numeric values and valid side.

        Args:
            fill: Fill event dict with keys: qty, price, side

        Returns:
            Error message string if validation fails, None if valid

        Note:
            - Rejects negative, zero, NaN, or infinite quantities/prices
            - Rejects non-numeric qty/price values
            - Side must be "buy" or "sell" (case-insensitive)
            - Used by on_fill() to prevent position corruption
        """
        # Validate qty
        try:
            qty = float(fill.get("qty", 0.0) or 0.0)
            if qty <= 0.0:
                return "qty must be positive"
            if not math.isfinite(qty):
                return "qty must be finite (not NaN or Inf)"
        except (TypeError, ValueError) as e:
            return f"invalid qty type: {e}"

        # Validate price
        try:
            price = float(fill.get("price", 0.0) or 0.0)
            if price <= 0.0:
                return "price must be positive"
            if not math.isfinite(price):
                return "price must be finite (not NaN or Inf)"
        except (TypeError, ValueError) as e:
            return f"invalid price type: {e}"

        # Validate side
        side = str(fill.get("side", "") or "").lower()
        if side not in ("buy", "sell"):
            return f"invalid side: {side} (must be buy or sell)"

        return None  # Valid

    def update_unrealized_pnl(
        self,
        symbol_state: SymbolState,
        mark_price: float,
        est_exit_commission: float = 0.0,
    ) -> None:
        """
        Update unrealized PnL for open position.

        Calculates current unrealized profit/loss based on the difference between
        current market price and average entry price, accounting for position direction.

        Args:
            symbol_state: Symbol state with position to update
            mark_price: Current market price for PnL calculation
            est_exit_commission: Optional estimated exit commission to subtract.
                                 M6 fix: For more accurate PnL, pass qty * commission_per_share.

        Side Effects:
            Updates symbol_state.position.unrealized_pnl in-place

        Note:
            - For LONG: unrealized = (current - avg_entry) * qty - est_exit_commission
            - For SHORT: unrealized = (avg_entry - current) * qty - est_exit_commission
            - Called best-effort; errors silently ignored
            - M6: By default shows gross unrealized PnL. Pass est_exit_commission
              for net unrealized that accounts for exit costs.
        """
        pos = symbol_state.position
        if pos is None:
            return
        # H3 fix: Use enum comparison for type safety
        if pos.side == Side.LONG:
            gross_pnl = (float(mark_price) - float(pos.avg_price)) * float(pos.qty)
        else:  # Side.SHORT
            gross_pnl = (float(pos.avg_price) - float(mark_price)) * float(pos.qty)

        # M6 fix: Subtract estimated exit commission for more accurate unrealized PnL
        pos.unrealized_pnl = gross_pnl - float(est_exit_commission)

    def on_fill(
        self,
        symbol_state: SymbolState,
        market_state: MarketState,
        fill: Dict[str, Any],
        *,
        risk_cfg: Optional[Dict[str, Any]] = None,
    ) -> FillDecision:
        """
        Process a fill and update position state.

        Handles both position increases (adding to position) and decreases
        (partial or full exits). Calculates realized PnL for exits and updates
        unrealized PnL for the remaining position. Uses FIFO cost basis for
        partial exit PnL calculation.

        Args:
            symbol_state: Current symbol state containing position information
            market_state: Current market regime and state for trade recording
            fill: Fill event dict with keys: symbol, qty, price, side, timestamp, correlation_id
            risk_cfg: Optional risk config with commission_per_share, min_commission, slippage_bps

        Returns:
            FillDecision containing:
                - event: "increased", "decreased", "closed", "opened", or "ignored"
                - realized_pnl_delta: $ PnL from this fill (0 for increases)
                - close_qty: Quantity that reduced position (0 for increases)
                - closed_trade: ClosedTradeInfo if position fully closed, else None
        """
        # Validate fill data first to prevent position corruption
        validation_error = self._validate_fill(fill)
        if validation_error:
            return FillDecision(
                event="ignored",
                realized_pnl_delta=0.0,
                close_qty=0.0,
                closed_trade=None,
            )

        # Extract fill data (validated above)
        fill_data = self._extract_fill_data(fill, symbol_state, market_state)
        cfg = self._extract_risk_config(risk_cfg)

        # Case 1: No existing position - open new position
        if symbol_state.position is None:
            return self._open_new_position(
                symbol_state, market_state, fill, fill_data, cfg
            )

        pos = symbol_state.position
        is_same_side = (pos.side == Side.LONG and fill_data["side"] == "buy") or (
            pos.side == Side.SHORT and fill_data["side"] == "sell"
        )

        # Case 2: Same side fill - increase position
        if is_same_side:
            return self._increase_position(symbol_state, fill_data)

        # Case 3: Opposite side fill - reduce or close position
        return self._reduce_or_close_position(
            symbol_state, market_state, fill_data, cfg
        )

    def _extract_fill_data(
        self,
        fill: Dict[str, Any],
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Dict[str, Any]:
        """Extract and normalize fill data into a standard dict."""
        return {
            "symbol": str(fill.get("symbol", "") or symbol_state.symbol),
            "qty": float(fill.get("qty", 0.0) or 0.0),
            "price": float(fill.get("price", 0.0) or 0.0),
            "side": str(fill.get("side", "") or "").lower(),
            "timestamp": fill.get("timestamp") or market_state.time,
            "correlation_id": str(fill.get("correlation_id", "") or ""),
            "strategy": fill.get("strategy", "unknown"),
        }

    def _extract_risk_config(
        self, risk_cfg: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Extract and normalize risk configuration."""
        cfg = risk_cfg if isinstance(risk_cfg, dict) else {}
        return {
            "commission_per_share": float(cfg.get("commission_per_share", 0.0) or 0.0),
            "min_commission": float(cfg.get("min_commission", 0.0) or 0.0),
            "slippage_bps": float(cfg.get("slippage_bps", 0.0) or 0.0),
        }

    def _get_entry_context(
        self, symbol_state: SymbolState, correlation_id: str, fill: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], Any, str]:
        """Get entry context from pending entries or fill data."""
        entry_ctx = None
        pending = symbol_state.meta.get("pending_entries")
        if correlation_id and isinstance(pending, dict):
            entry_ctx = pending.pop(correlation_id, None)

        entry_time = (
            entry_ctx.get("entry_time")
            if isinstance(entry_ctx, dict) and entry_ctx.get("entry_time")
            else fill.get("timestamp")
        )
        strategy = (
            entry_ctx.get("strategy") if isinstance(entry_ctx, dict) else None
        ) or fill.get("strategy", "unknown")

        return entry_ctx, entry_time, strategy

    def _apply_costs_to_position(
        self,
        pos: Position,
        qty: float,
        price: float,
        cfg: Dict[str, float],
    ) -> None:
        """Apply commission and slippage costs to position."""
        cps = cfg["commission_per_share"]
        min_c = cfg["min_commission"]
        slippage_bps = cfg["slippage_bps"]

        if cps > 0.0 or min_c > 0.0:
            pos.commission = float(pos.commission or 0.0) + float(
                max(min_c, cps * float(qty))
            )
        if slippage_bps > 0.0:
            pos.slippage_estimate = float(pos.slippage_estimate or 0.0) + float(
                (slippage_bps / 10000.0) * float(qty) * float(price)
            )

    def _open_new_position(
        self,
        symbol_state: SymbolState,
        market_state: MarketState,
        fill: Dict[str, Any],
        fill_data: Dict[str, Any],
        cfg: Dict[str, float],
    ) -> FillDecision:
        """Open a new position from a fill."""
        side = Side.LONG if fill_data["side"] == "buy" else Side.SHORT
        entry_ctx, entry_time, strategy = self._get_entry_context(
            symbol_state, fill_data["correlation_id"], fill
        )

        symbol_state.position = Position(
            symbol=fill_data["symbol"],
            side=side,
            qty=fill_data["qty"],
            avg_price=fill_data["price"],
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            strategy=str(strategy),
            entry_time=entry_time,
            correlation_id=fill_data["correlation_id"],
            regime_at_entry=market_state.regime,
            open_risk=(
                safe_float(entry_ctx.get("open_risk"))
                if isinstance(entry_ctx, dict)
                else None
            ),
            stop_price=(
                safe_float(entry_ctx.get("stop_price"))
                if isinstance(entry_ctx, dict)
                else None
            ),
            target_price=(
                safe_float(entry_ctx.get("target_price"))
                if isinstance(entry_ctx, dict)
                else None
            ),
            entry_features=(
                entry_ctx.get("features") if isinstance(entry_ctx, dict) else None
            ),
            mae_r=0.0,
            mfe_r=0.0,
            commission=0.0,
            slippage_estimate=0.0,
            max_hold_seconds=(
                safe_int(entry_ctx.get("max_hold_seconds"))
                if isinstance(entry_ctx, dict)
                else None
            ),
        )

        self._apply_costs_to_position(
            symbol_state.position, fill_data["qty"], fill_data["price"], cfg
        )

        try:
            self.update_unrealized_pnl(symbol_state, float(fill_data["price"]))
        except Exception:
            pass

        # H1 fix: Mark position as updated to prevent reconciliation races
        symbol_state.position.last_updated = datetime.now(timezone.utc)

        return FillDecision(
            event="opened",
            realized_pnl_delta=0.0,
            close_qty=0.0,
            closed_trade=None,
        )

    def _increase_position(
        self, symbol_state: SymbolState, fill_data: Dict[str, Any]
    ) -> FillDecision:
        """Increase an existing position with a same-side fill."""
        pos = symbol_state.position
        assert pos is not None, "Position must exist for increase"
        total_cost = (pos.qty * pos.avg_price) + (fill_data["qty"] * fill_data["price"])
        total_qty = pos.qty + fill_data["qty"]
        if total_qty > 0:
            pos.avg_price = total_cost / total_qty
            pos.qty = total_qty
        try:
            self.update_unrealized_pnl(symbol_state, float(fill_data["price"]))
        except Exception:
            pass
        return FillDecision(
            event="increased",
            realized_pnl_delta=0.0,
            close_qty=0.0,
            closed_trade=None,
        )

    def _calculate_pnl(
        self, pos: Position, fill_price: float, close_qty: float
    ) -> float:
        """Calculate realized PnL for a position reduction."""
        if pos.side == Side.LONG:
            return (fill_price - float(pos.avg_price)) * float(close_qty)
        else:
            return (float(pos.avg_price) - fill_price) * float(close_qty)

    def _build_closed_trade_info(
        self,
        pos: Position,
        fill_data: Dict[str, Any],
        market_state: MarketState,
        pnl: float,
        close_qty: float,
    ) -> ClosedTradeInfo:
        """Build ClosedTradeInfo for a fully closed position."""
        entry_time_final = pos.entry_time or fill_data["timestamp"]
        exit_time = fill_data["timestamp"]

        holding_period_seconds = None
        try:
            holding_period_seconds = (exit_time - entry_time_final).total_seconds()
        except Exception:
            pass

        regime_at_entry = (
            pos.regime_at_entry.value
            if isinstance(pos.regime_at_entry, Regime)
            else str(
                getattr(pos.regime_at_entry, "value", pos.regime_at_entry or "unknown")
            )
        )
        regime_at_exit = (
            market_state.regime.value
            if isinstance(market_state.regime, Regime)
            else str(getattr(market_state.regime, "value", market_state.regime))
        )

        pnl_net = (
            float(pnl)
            - float(pos.commission or 0.0)
            - float(pos.slippage_estimate or 0.0)
        )

        # H2 fix: Safe R-multiple calculation
        pnl_r = None
        if pos.open_risk is not None and float(pos.open_risk) != 0.0:
            pnl_r = float(pnl) / float(pos.open_risk)

        return ClosedTradeInfo(
            symbol=pos.symbol,
            strategy=pos.strategy,
            regime_at_entry=regime_at_entry or "unknown",
            regime_at_exit=regime_at_exit or "unknown",
            side=pos.side.value,
            qty=float(close_qty),
            entry_time=entry_time_final,
            exit_time=exit_time,
            entry_price=float(pos.avg_price),
            exit_price=float(fill_data["price"]),
            pnl_gross=float(pnl),
            pnl_net=float(pnl_net),
            initial_risk=pos.open_risk,
            mae_r=float(pos.mae_r),
            mfe_r=float(pos.mfe_r),
            commission=float(pos.commission or 0.0),
            slippage_estimate=float(pos.slippage_estimate or 0.0),
            pnl_r=pnl_r,
            holding_period_seconds=holding_period_seconds,
            features_json=pos.entry_features,
            correlation_id=str(pos.correlation_id or fill_data["correlation_id"]),
        )

    def _reduce_or_close_position(
        self,
        symbol_state: SymbolState,
        market_state: MarketState,
        fill_data: Dict[str, Any],
        cfg: Dict[str, float],
    ) -> FillDecision:
        """Reduce or close an existing position with opposite-side fill."""
        pos = symbol_state.position
        assert pos is not None, "Position must exist for reduce/close"
        close_qty = min(float(pos.qty), fill_data["qty"])
        pnl = self._calculate_pnl(pos, fill_data["price"], close_qty)

        pos.realized_pnl = float(pos.realized_pnl) + float(pnl)
        self._apply_costs_to_position(pos, close_qty, fill_data["price"], cfg)
        pos.qty = float(pos.qty) - float(close_qty)

        # H1 fix: Mark position as updated to prevent reconciliation races
        pos.last_updated = datetime.now(timezone.utc)

        if pos.qty > 0:
            try:
                self.update_unrealized_pnl(symbol_state, float(fill_data["price"]))
            except Exception:
                pass
            return FillDecision(
                event="reduced",
                realized_pnl_delta=float(pnl),
                close_qty=float(close_qty),
                closed_trade=None,
            )

        # Position fully closed
        closed = self._build_closed_trade_info(
            pos, fill_data, market_state, pnl, close_qty
        )
        symbol_state.position = None
        return FillDecision(
            event="closed",
            realized_pnl_delta=float(pnl),
            close_qty=float(close_qty),
            closed_trade=closed,
        )

    def on_bar(
        self,
        symbol_state: SymbolState,
        market_state: MarketState,
        *,
        broker_managed_exits: bool = False,
    ) -> ExitDecision:
        """
        Check if position should be closed based on stop loss or take profit.

        Evaluates Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
        against position stop loss and take profit levels. Only checks exit crossings
        when exits are NOT delegated to broker (i.e., when bracket orders are not used).

        Args:
            symbol_state: Current symbol state with position and entry context
            market_state: Current market state, including time and regime
            broker_managed_exits: If True, broker handles exits (e.g., bracket orders),
                                  so this method should not generate exit intents.

        Returns:
            ExitDecision containing:
                - intent: OrderIntent to close position if exit triggered, else None
                - reason: Exit reason if triggered, else None
        """
        pos = symbol_state.position
        if pos is None:
            return ExitDecision(intent=None, reason=None)

        last_bar = symbol_state.bars[-1] if symbol_state.bars else None
        if last_bar is None:
            return ExitDecision(intent=None, reason=None)

        # Track MAE/MFE (best-effort)
        self._update_mae_mfe(pos, last_bar)

        # Check max-hold exit (independent of broker delegation)
        max_hold_exit = self._check_max_hold_exit(pos, market_state)
        if max_hold_exit:
            return max_hold_exit

        # PRD 6.7: skip stop/target when broker manages exits
        if broker_managed_exits:
            return ExitDecision(intent=None, reason=None)

        # Check stop/target exit
        return self._check_stop_target_exit(pos, last_bar)

    def _update_mae_mfe(self, pos: Position, last_bar: Any) -> None:
        """Update MAE/MFE tracking for a position (best-effort)."""
        try:
            open_risk = pos.open_risk
            if open_risk is None or open_risk == 0.0 or pos.qty <= 0:
                return
            risk_per_share = float(open_risk) / float(pos.qty)
            if risk_per_share <= 0:
                return

            if pos.side == Side.LONG:
                adverse_r = abs(
                    min(0.0, (last_bar.low - pos.avg_price) / risk_per_share)
                )
                favorable_r = max(0.0, (last_bar.high - pos.avg_price) / risk_per_share)
            else:
                adverse_r = abs(
                    min(0.0, (pos.avg_price - last_bar.high) / risk_per_share)
                )
                favorable_r = max(0.0, (pos.avg_price - last_bar.low) / risk_per_share)
            pos.mae_r = max(pos.mae_r, adverse_r)
            pos.mfe_r = max(pos.mfe_r, favorable_r)
        except Exception:
            _logger.debug("MAE/MFE update failed", exc_info=True)

    def _check_max_hold_exit(
        self, pos: Position, market_state: MarketState
    ) -> Optional[ExitDecision]:
        """Check if position exceeds max hold time."""
        try:
            if (
                pos.max_hold_seconds is None
                or pos.entry_time is None
                or market_state.time is None
            ):
                return None
            held = (market_state.time - pos.entry_time).total_seconds()
            if held < float(pos.max_hold_seconds):
                return None
            return self._create_exit_intent(pos, "MAX_HOLD_EXCEEDED")
        except Exception:
            _logger.debug("Max-hold check failed", exc_info=True)
            return None

    def _check_stop_target_exit(self, pos: Position, last_bar: Any) -> ExitDecision:
        """Check if stop or target was hit on this bar."""
        stop_price = pos.stop_price
        target_price = pos.target_price

        if stop_price is None and target_price is None:
            return ExitDecision(intent=None, reason=None)

        hit_stop = False
        hit_target = False

        if pos.side == Side.LONG:
            if stop_price is not None and last_bar.low <= stop_price:
                hit_stop = True
            if target_price is not None and last_bar.high >= target_price:
                hit_target = True
        else:
            if stop_price is not None and last_bar.high >= stop_price:
                hit_stop = True
            if target_price is not None and last_bar.low <= target_price:
                hit_target = True

        if not (hit_stop or hit_target):
            return ExitDecision(intent=None, reason=None)

        # M3 fix: Priority TARGET > STOP (more favorable for trader)
        reason = "TARGET_HIT" if hit_target else "STOP_HIT"
        return self._create_exit_intent(pos, reason)

    def _create_exit_intent(self, pos: Position, reason: str) -> ExitDecision:
        """Create an exit intent for a given reason."""
        exit_side = OrderSide.SELL if pos.side == Side.LONG else OrderSide.BUY
        intent = OrderIntent(
            symbol=pos.symbol,
            side=exit_side,
            qty=pos.qty,
            order_type=OrderType.MARKET,
            limit_price=None,
            time_in_force="day",
            correlation_id=pos.correlation_id,
            strategy=pos.strategy,
            stop_loss=None,
            take_profit=None,
            meta={"exit_reason": reason},
        )
        return ExitDecision(intent=intent, reason=reason)
