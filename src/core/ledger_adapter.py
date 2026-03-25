"""CerberusLedgerAdapter — maps Cerberus trade lifecycle to empire_core.ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog
from empire_core.ledger import LedgerWriter

_logger = structlog.get_logger(__name__)


def _to_python_datetime(val: Any) -> datetime | None:
    """Convert pandas Timestamp or other datetime-like to Python datetime for SQLite."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    # pandas Timestamp has .to_pydatetime()
    if hasattr(val, "to_pydatetime"):
        return val.to_pydatetime()
    # Nanosecond epoch integer (e.g. 1672758300000000000) — convert to UTC datetime
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(val / 1e9, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return datetime.fromtimestamp(val, tz=timezone.utc)
    return val


class CerberusLedgerAdapter:
    """Wraps LedgerWriter to record Cerberus trades in the unified ledger.

    Tracks open trades by symbol so that close events can reference the
    correct ledger trade_id.  If a trade is closed that was never tracked
    (e.g. opened before the adapter was initialised), the adapter creates
    both the open and close records back-to-back.
    """

    def __init__(self, ledger: LedgerWriter) -> None:
        self._ledger = ledger
        # symbol -> ledger trade_id
        self._open_trades: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Hook: position opened
    # ------------------------------------------------------------------

    def on_trade_opened(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        entry_time: datetime,
        strategy: str,
        correlation_id: str,
        regime_tags: Optional[Dict[str, str]] = None,
        initial_risk: Optional[float] = None,
    ) -> Optional[str]:
        """Record a new trade in the ledger and track it internally.

        Returns the ledger trade_id, or None if recording failed.
        """
        ledger_symbol = f"equity:{symbol}"
        regime_entry = (regime_tags or {}).get("trend")
        tags = _regime_tags_to_list(regime_tags)

        try:
            trade_id = self._ledger.open_trade(
                correlation_id=correlation_id,
                symbol=ledger_symbol,
                side=side,
                qty=qty,
                entry_price=entry_price,
                entry_time=entry_time,
                strategy=strategy,
                initial_risk=initial_risk,
                regime_entry=regime_entry,
                tags=tags,
                meta={"regime_tags": regime_tags} if regime_tags else None,
            )
            self._open_trades[symbol] = trade_id
            _logger.debug(
                "ledger_trade_opened",
                symbol=ledger_symbol,
                trade_id=trade_id,
                strategy=strategy,
            )
            return trade_id
        except Exception:
            _logger.error("ledger_open_trade_failed — audit trail incomplete", symbol=symbol, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Hook: position closed
    # ------------------------------------------------------------------

    def on_trade_closed(self, closed: Any) -> None:
        """Record a closed trade in the ledger.

        *closed* is expected to be a ``ClosedTradeInfo`` instance (imported
        at the call site to avoid circular imports).

        If the trade was not previously tracked via ``on_trade_opened``,
        both an open and close record are created.
        """
        symbol = closed.symbol
        ledger_symbol = f"equity:{symbol}"
        trade_id = self._open_trades.pop(symbol, None)

        regime_exit = (closed.regime_tags_at_exit or {}).get("trend")
        exit_tags = _regime_tags_to_list(closed.regime_tags_at_exit)

        holding_seconds: int | None = None
        if closed.holding_period_seconds is not None:
            holding_seconds = int(closed.holding_period_seconds)

        meta: Dict[str, Any] = {}
        if closed.regime_tags_at_exit:
            meta["regime_tags_exit"] = closed.regime_tags_at_exit
        if closed.features_json:
            meta["features"] = closed.features_json

        try:
            # Back-fill open record for trades that started before the adapter
            if trade_id is None:
                regime_entry = (closed.regime_tags_at_entry or {}).get("trend")
                entry_tags = _regime_tags_to_list(closed.regime_tags_at_entry)
                entry_meta: Dict[str, Any] | None = (
                    {"regime_tags": closed.regime_tags_at_entry} if closed.regime_tags_at_entry else None
                )
                trade_id = self._ledger.open_trade(
                    correlation_id=closed.correlation_id,
                    symbol=ledger_symbol,
                    side=closed.side,
                    qty=closed.qty,
                    entry_price=closed.entry_price,
                    entry_time=_to_python_datetime(closed.entry_time),
                    strategy=closed.strategy,
                    initial_risk=closed.initial_risk,
                    regime_entry=regime_entry,
                    tags=entry_tags,
                    meta=entry_meta,
                )
                _logger.debug(
                    "ledger_backfill_open",
                    symbol=ledger_symbol,
                    trade_id=trade_id,
                )

            self._ledger.close_trade(
                trade_id=trade_id,
                exit_price=closed.exit_price,
                exit_time=_to_python_datetime(closed.exit_time),
                pnl_gross=closed.pnl_gross,
                pnl_net=closed.pnl_net,
                commission_total=closed.commission,
                pnl_r=closed.pnl_r,
                mae_r=closed.mae_r,
                mfe_r=closed.mfe_r,
                slippage=closed.slippage_estimate,
                regime_exit=regime_exit,
                holding_seconds=holding_seconds,
                tags=exit_tags,
                meta=meta if meta else None,
            )
            _logger.debug(
                "ledger_trade_closed",
                symbol=ledger_symbol,
                trade_id=trade_id,
                pnl_net=closed.pnl_net,
            )
        except Exception:
            _logger.error("ledger_close_trade_failed — audit trail incomplete", symbol=symbol, exc_info=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _regime_tags_to_list(regime_tags: Optional[Dict[str, str]]) -> list[str] | None:
    """Convert regime tags dict to a flat list for the ledger tags field.

    Example: {"trend": "bullish", "vol": "high"} -> ["trend:bullish", "vol:high"]
    """
    if not regime_tags:
        return None
    return [f"{k}:{v}" for k, v in sorted(regime_tags.items())]
