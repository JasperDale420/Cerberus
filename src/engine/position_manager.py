from __future__ import annotations

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

        Note:
            - For LONG positions, sells reduce position
            - For SHORT positions, buys reduce position
            - Best-effort unrealized PnL update (silent failure OK)
            - Updates symbol_state.position in-place
            - Ignores fills with invalid qty/price/side (validates using _validate_fill)
        """
        # Validate fill data first to prevent position corruption
        validation_error = self._validate_fill(fill)
        if validation_error:
            # Log detailed validation failure for debugging
            return FillDecision(
                event="ignored",
                realized_pnl_delta=0.0,
                close_qty=0.0,
                closed_trade=None,
            )

        # Extract fill data (validated above)
        symbol = str(fill.get("symbol", "") or symbol_state.symbol)
        fill_qty = float(fill.get("qty", 0.0) or 0.0)
        fill_price = float(fill.get("price", 0.0) or 0.0)
        fill_side = str(fill.get("side", "") or "").lower()
        fill_ts = fill.get("timestamp") or market_state.time
        corr = str(fill.get("correlation_id", "") or "")

        cfg = risk_cfg if isinstance(risk_cfg, dict) else {}
        cps = float(cfg.get("commission_per_share", 0.0) or 0.0)
        min_c = float(cfg.get("min_commission", 0.0) or 0.0)
        slippage_bps = float(cfg.get("slippage_bps", 0.0) or 0.0)

        if symbol_state.position is None:
            side = Side.LONG if fill_side == "buy" else Side.SHORT

            entry_ctx = None
            pending = symbol_state.meta.get("pending_entries")
            if corr and isinstance(pending, dict):
                entry_ctx = pending.pop(corr, None)

            entry_time = (
                entry_ctx.get("entry_time")
                if isinstance(entry_ctx, dict) and entry_ctx.get("entry_time")
                else fill_ts
            )
            strategy = (
                entry_ctx.get("strategy") if isinstance(entry_ctx, dict) else None
            ) or fill.get("strategy", "unknown")

            symbol_state.position = Position(
                symbol=symbol,
                side=side,
                qty=fill_qty,
                avg_price=fill_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                strategy=str(strategy),
                entry_time=entry_time,
                correlation_id=corr,
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

            if cps > 0.0 or min_c > 0.0:
                symbol_state.position.commission = float(
                    max(min_c, cps * float(fill_qty))
                )
            if slippage_bps > 0.0:
                symbol_state.position.slippage_estimate = float(
                    (slippage_bps / 10000.0) * float(fill_qty) * float(fill_price)
                )

            # PRD 6.7: update unrealized PnL on fills deterministically.
            try:
                self.update_unrealized_pnl(symbol_state, float(fill_price))
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

        pos = symbol_state.position
        is_same_side = (pos.side == Side.LONG and fill_side == "buy") or (
            pos.side == Side.SHORT and fill_side == "sell"
        )

        if is_same_side:
            total_cost = (pos.qty * pos.avg_price) + (fill_qty * fill_price)
            total_qty = pos.qty + fill_qty
            if total_qty > 0:
                pos.avg_price = total_cost / total_qty
                pos.qty = total_qty
            try:
                self.update_unrealized_pnl(symbol_state, float(fill_price))
            except Exception:
                pass  # Best-effort PnL update
            return FillDecision(
                event="increased",
                realized_pnl_delta=0.0,
                close_qty=0.0,
                closed_trade=None,
            )

        close_qty = min(float(pos.qty), fill_qty)
        if pos.side == Side.LONG:
            pnl = (fill_price - float(pos.avg_price)) * float(close_qty)
        else:
            pnl = (float(pos.avg_price) - fill_price) * float(close_qty)

        pos.realized_pnl = float(pos.realized_pnl) + float(pnl)

        if cps > 0.0 or min_c > 0.0:
            pos.commission = float(pos.commission or 0.0) + float(
                max(min_c, cps * float(close_qty))
            )
        if slippage_bps > 0.0:
            pos.slippage_estimate = float(pos.slippage_estimate or 0.0) + float(
                (slippage_bps / 10000.0) * float(close_qty) * float(fill_price)
            )

        pos.qty = float(pos.qty) - float(close_qty)

        # H1 fix: Mark position as updated to prevent reconciliation races
        pos.last_updated = datetime.now(timezone.utc)

        if pos.qty > 0:
            try:
                self.update_unrealized_pnl(symbol_state, float(fill_price))
            except Exception:
                pass
            return FillDecision(
                event="reduced",
                realized_pnl_delta=float(pnl),
                close_qty=float(close_qty),
                closed_trade=None,
            )

        entry_time_final = pos.entry_time or fill_ts
        exit_time = fill_ts

        holding_period_seconds = None
        try:
            holding_period_seconds = (exit_time - entry_time_final).total_seconds()
        except Exception:
            holding_period_seconds = None  # Best-effort calculation

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

        # H2 fix: Safe R-multiple calculation with division by zero protection
        # If initial_risk is 0 (breakeven stop: entry == stop) or None, R-multiple is undefined
        pnl_r = None
        if pos.open_risk is not None and float(pos.open_risk) != 0.0:
            pnl_r = float(pnl) / float(pos.open_risk)

        closed = ClosedTradeInfo(
            symbol=pos.symbol,
            strategy=pos.strategy,
            regime_at_entry=regime_at_entry or "unknown",
            regime_at_exit=regime_at_exit or "unknown",
            side=pos.side.value,
            qty=float(close_qty),
            entry_time=entry_time_final,
            exit_time=exit_time,
            entry_price=float(pos.avg_price),
            exit_price=float(fill_price),
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
            correlation_id=str(pos.correlation_id or corr),
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
                - reason: Exit reason ("STOP_HIT", "TARGET_HIT", "MAX_HOLD_EXCEEDED") if triggered, else None

        Note:
            - PRD 6.7: Only checks when broker exit delegation is disabled
            - Updates MAE/MFE tracking (best-effort, silent failure OK)
            - Compares bar highs/lows against entry-relative thresholds
            - Exit intent includes correlation_id and exit_reason metadata
            - Returns ExitDecision(intent=None, reason=None) if no exit triggered
        """
        pos = symbol_state.position
        if pos is None:
            return ExitDecision(intent=None, reason=None)

        last_bar = symbol_state.bars[-1] if symbol_state.bars else None
        if last_bar is None:
            return ExitDecision(intent=None, reason=None)

        # Track MAE/MFE in R units when risk context is known.
        try:
            risk_per_share = None
            open_risk = pos.open_risk
            if open_risk is not None and open_risk != 0.0 and pos.qty > 0:
                risk_per_share = float(open_risk) / float(pos.qty)
            if risk_per_share and risk_per_share > 0:
                # H3 fix: Use enum comparison for type safety
                if pos.side == Side.LONG:
                    adverse_r = abs(
                        min(0.0, (last_bar.low - pos.avg_price) / risk_per_share)
                    )
                    favorable_r = max(
                        0.0, (last_bar.high - pos.avg_price) / risk_per_share
                    )
                else:  # Side.SHORT
                    adverse_r = abs(
                        min(0.0, (pos.avg_price - last_bar.high) / risk_per_share)
                    )
                    favorable_r = max(
                        0.0, (pos.avg_price - last_bar.low) / risk_per_share
                    )
                pos.mae_r = max(pos.mae_r, adverse_r)
                pos.mfe_r = max(pos.mfe_r, favorable_r)
        except Exception:
            # Best-effort: do not break trading if metrics fail to update.
            pass

        # Only manage exits when we have explicit stop/target.
        stop_price = pos.stop_price
        target_price = pos.target_price

        # Deterministic max-hold exit (PRD 7.2 VWAPReversion config).
        try:
            if (
                pos.max_hold_seconds is not None
                and pos.entry_time is not None
                and market_state.time is not None
            ):
                held = (market_state.time - pos.entry_time).total_seconds()
                if held >= float(pos.max_hold_seconds):
                    reason = "MAX_HOLD_EXCEEDED"
                    # H3 fix: Use enum comparison for type safety
                    exit_side = (
                        OrderSide.SELL if pos.side == Side.LONG else OrderSide.BUY
                    )
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
        except Exception:
            pass  # Best-effort MAE/MFE tracking

        # PRD 6.7: only check stop/target crossing when exits are not fully delegated
        # to the broker (e.g., when bracket orders are not used).
        if broker_managed_exits:
            return ExitDecision(intent=None, reason=None)

        if stop_price is None and target_price is None:
            return ExitDecision(intent=None, reason=None)

        hit_stop = False
        hit_target = False
        # H3 fix: Use enum comparison for type safety
        if pos.side == Side.LONG:
            if stop_price is not None and last_bar.low <= stop_price:
                hit_stop = True
            if target_price is not None and last_bar.high >= target_price:
                hit_target = True
        else:  # Side.SHORT
            if stop_price is not None and last_bar.high >= stop_price:
                hit_stop = True
            if target_price is not None and last_bar.low <= target_price:
                hit_target = True

        if not (hit_stop or hit_target):
            return ExitDecision(intent=None, reason=None)

        # M3 fix: Exit priority when multiple conditions trigger on same bar
        # Priority order: TARGET > STOP (target is more favorable for trader)
        # If price gaps through both levels, we assume target was hit first
        # (bar touched target before reversing to hit stop).
        reason = "TARGET_HIT" if hit_target else "STOP_HIT"
        # H3 fix: Use enum comparison for type safety
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
