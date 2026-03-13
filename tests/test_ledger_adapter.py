"""Tests for CerberusLedgerAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.core.ledger_adapter import CerberusLedgerAdapter, _regime_tags_to_list
from src.engine.position_manager import ClosedTradeInfo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ledger():
    ledger = MagicMock()
    ledger.open_trade.return_value = "trade-001"
    return ledger


@pytest.fixture
def adapter(mock_ledger):
    return CerberusLedgerAdapter(mock_ledger)


def _make_closed_trade(**overrides) -> ClosedTradeInfo:
    defaults = dict(
        symbol="AAPL",
        strategy="vwap_reversion",
        regime_tags_at_entry={"trend": "bullish", "vol": "low"},
        regime_tags_at_exit={"trend": "bearish", "vol": "high"},
        side="long",
        qty=100.0,
        entry_time=datetime(2026, 3, 13, 14, 30, tzinfo=timezone.utc),
        exit_time=datetime(2026, 3, 13, 15, 0, tzinfo=timezone.utc),
        entry_price=150.0,
        exit_price=152.0,
        pnl_gross=200.0,
        pnl_net=195.0,
        initial_risk=50.0,
        mae_r=0.3,
        mfe_r=4.0,
        commission=3.0,
        slippage_estimate=2.0,
        pnl_r=4.0,
        holding_period_seconds=1800.0,
        features_json={"rsi": 45.0},
        correlation_id="corr-abc",
    )
    defaults.update(overrides)
    return ClosedTradeInfo(**defaults)


# ---------------------------------------------------------------------------
# _regime_tags_to_list helper
# ---------------------------------------------------------------------------


class TestRegimeTagsToList:
    def test_none(self):
        assert _regime_tags_to_list(None) is None

    def test_empty(self):
        assert _regime_tags_to_list({}) is None

    def test_converts_sorted(self):
        result = _regime_tags_to_list({"vol": "high", "trend": "bullish"})
        assert result == ["trend:bullish", "vol:high"]


# ---------------------------------------------------------------------------
# on_trade_opened
# ---------------------------------------------------------------------------


class TestOnTradeOpened:
    def test_records_open_trade(self, adapter, mock_ledger):
        entry_time = datetime(2026, 3, 13, 14, 30, tzinfo=timezone.utc)
        regime = {"trend": "bullish", "vol": "low"}

        trade_id = adapter.on_trade_opened(
            symbol="AAPL",
            side="long",
            qty=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            strategy="vwap_reversion",
            correlation_id="corr-123",
            regime_tags=regime,
            initial_risk=50.0,
        )

        assert trade_id == "trade-001"
        mock_ledger.open_trade.assert_called_once_with(
            correlation_id="corr-123",
            symbol="equity:AAPL",
            side="long",
            qty=100.0,
            entry_price=150.0,
            entry_time=entry_time,
            strategy="vwap_reversion",
            initial_risk=50.0,
            regime_entry="bullish",
            tags=["trend:bullish", "vol:low"],
            meta={"regime_tags": regime},
        )

    def test_tracks_open_trade_by_symbol(self, adapter, mock_ledger):
        adapter.on_trade_opened(
            symbol="TSLA",
            side="long",
            qty=50,
            entry_price=200.0,
            entry_time=datetime.now(timezone.utc),
            strategy="orb",
            correlation_id="c1",
        )
        assert "TSLA" in adapter._open_trades

    def test_no_regime_tags(self, adapter, mock_ledger):
        adapter.on_trade_opened(
            symbol="MSFT",
            side="short",
            qty=10,
            entry_price=300.0,
            entry_time=datetime.now(timezone.utc),
            strategy="gap_fill",
            correlation_id="c2",
            regime_tags=None,
        )
        call_kwargs = mock_ledger.open_trade.call_args.kwargs
        assert call_kwargs["regime_entry"] is None
        assert call_kwargs["tags"] is None
        assert call_kwargs["meta"] is None

    def test_exception_returns_none(self, adapter, mock_ledger):
        mock_ledger.open_trade.side_effect = RuntimeError("db error")
        result = adapter.on_trade_opened(
            symbol="AAPL",
            side="long",
            qty=10,
            entry_price=100.0,
            entry_time=datetime.now(timezone.utc),
            strategy="s",
            correlation_id="c",
        )
        assert result is None


# ---------------------------------------------------------------------------
# on_trade_closed
# ---------------------------------------------------------------------------


class TestOnTradeClosed:
    def test_close_tracked_trade(self, adapter, mock_ledger):
        """Close a trade that was previously opened via on_trade_opened."""
        adapter._open_trades["AAPL"] = "trade-001"
        closed = _make_closed_trade()

        adapter.on_trade_closed(closed)

        # Should NOT call open_trade again (already tracked)
        mock_ledger.open_trade.assert_not_called()

        mock_ledger.close_trade.assert_called_once()
        kw = mock_ledger.close_trade.call_args.kwargs
        assert kw["trade_id"] == "trade-001"
        assert kw["exit_price"] == 152.0
        assert kw["pnl_gross"] == 200.0
        assert kw["pnl_net"] == 195.0
        assert kw["commission_total"] == 3.0
        assert kw["pnl_r"] == 4.0
        assert kw["mae_r"] == 0.3
        assert kw["mfe_r"] == 4.0
        assert kw["slippage"] == 2.0
        assert kw["regime_exit"] == "bearish"
        assert kw["holding_seconds"] == 1800
        assert "trend:bearish" in kw["tags"]

    def test_close_untracked_trade_backfills_open(self, adapter, mock_ledger):
        """Trade closed that was never tracked — should create open + close."""
        mock_ledger.open_trade.return_value = "backfill-001"
        closed = _make_closed_trade()

        adapter.on_trade_closed(closed)

        # Should create an open record first
        mock_ledger.open_trade.assert_called_once()
        open_kw = mock_ledger.open_trade.call_args.kwargs
        assert open_kw["symbol"] == "equity:AAPL"
        assert open_kw["correlation_id"] == "corr-abc"
        assert open_kw["regime_entry"] == "bullish"

        # Then close it
        mock_ledger.close_trade.assert_called_once()
        close_kw = mock_ledger.close_trade.call_args.kwargs
        assert close_kw["trade_id"] == "backfill-001"

    def test_symbol_removed_from_open_trades(self, adapter, mock_ledger):
        adapter._open_trades["AAPL"] = "trade-001"
        closed = _make_closed_trade()
        adapter.on_trade_closed(closed)
        assert "AAPL" not in adapter._open_trades

    def test_none_holding_period(self, adapter, mock_ledger):
        adapter._open_trades["AAPL"] = "trade-001"
        closed = _make_closed_trade(holding_period_seconds=None)
        adapter.on_trade_closed(closed)
        kw = mock_ledger.close_trade.call_args.kwargs
        assert kw["holding_seconds"] is None

    def test_no_features_json(self, adapter, mock_ledger):
        adapter._open_trades["AAPL"] = "trade-001"
        closed = _make_closed_trade(features_json=None, regime_tags_at_exit={})
        adapter.on_trade_closed(closed)
        kw = mock_ledger.close_trade.call_args.kwargs
        assert kw["meta"] is None

    def test_exception_does_not_propagate(self, adapter, mock_ledger):
        adapter._open_trades["AAPL"] = "trade-001"
        mock_ledger.close_trade.side_effect = RuntimeError("db error")
        closed = _make_closed_trade()
        # Should not raise
        adapter.on_trade_closed(closed)

    def test_close_with_no_regime_exit_tags(self, adapter, mock_ledger):
        adapter._open_trades["AAPL"] = "trade-001"
        closed = _make_closed_trade(regime_tags_at_exit={})
        adapter.on_trade_closed(closed)
        kw = mock_ledger.close_trade.call_args.kwargs
        assert kw["regime_exit"] is None
        assert kw["tags"] is None
