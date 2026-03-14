"""Unit tests for the StreamingScanner."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.data.client import StreamQuote, StreamTrade
from src.scanner.streaming_scanner import StreamingScanner

# ── Fixtures ──


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def mock_calculator():
    calc = MagicMock()
    calc.calculate_ofi = MagicMock(return_value=0.5)
    calc.calculate_tfi = MagicMock(return_value=-0.3)
    return calc


@pytest.fixture
def streaming_scanner(mock_calculator, mock_logger):
    return StreamingScanner(calculator=mock_calculator, logger=mock_logger)


# ── StreamingScanner Tests ──


class TestStreamingScanner:
    """Tests for streaming OFI/TFI incremental updates."""

    def test_add_remove_symbol(self, streaming_scanner):
        streaming_scanner.add_symbol("AAPL")
        assert "AAPL" in streaming_scanner.tracked_symbols

        streaming_scanner.remove_symbol("AAPL")
        assert "AAPL" not in streaming_scanner.tracked_symbols

    def test_get_features_default(self, streaming_scanner):
        feats = streaming_scanner.get_features("UNKNOWN")
        assert feats == {"ofi": 0.0, "tfi": 0.0, "high_low_spread": 0.0}

    @pytest.mark.asyncio
    async def test_on_quote_accumulates(self, streaming_scanner):
        """Quotes accumulate in the rolling window."""
        streaming_scanner.add_symbol("AAPL")
        now = datetime.now(timezone.utc)

        for i in range(3):
            quote = StreamQuote(
                symbol="AAPL",
                bid_price=150.0 + i,
                bid_size=100 + i * 10,
                ask_price=150.05 + i,
                ask_size=200,
                timestamp=now,
            )
            await streaming_scanner.on_quote("AAPL", quote)

        state = streaming_scanner._states["AAPL"]
        assert len(state.recent_quotes) == 3

    @pytest.mark.asyncio
    async def test_on_quote_triggers_ofi_calculation(self, streaming_scanner, mock_calculator):
        """OFI is recalculated when enough quotes are received and delta exceeds threshold."""
        streaming_scanner.add_symbol("AAPL")
        now = datetime.now(timezone.utc)

        # Set initial OFI to 0, calculator will return 0.5 -> delta of 0.5 > threshold
        for i in range(6):
            quote = StreamQuote(
                symbol="AAPL",
                bid_price=150.0 + i * 0.01,
                bid_size=100,
                ask_price=150.05,
                ask_size=200,
                timestamp=now,
            )
            await streaming_scanner.on_quote("AAPL", quote)

        mock_calculator.calculate_ofi.assert_called()
        state = streaming_scanner._states["AAPL"]
        assert state.last_ofi == 0.5

    @pytest.mark.asyncio
    async def test_on_trade_accumulates(self, streaming_scanner):
        """Trades accumulate in the rolling window."""
        streaming_scanner.add_symbol("TSLA")
        now = datetime.now(timezone.utc)

        for i in range(5):
            trade = StreamTrade(
                symbol="TSLA",
                price=200.0 + i,
                size=10,
                timestamp=now,
            )
            await streaming_scanner.on_trade("TSLA", trade)

        state = streaming_scanner._states["TSLA"]
        assert len(state.recent_trades) == 5

    @pytest.mark.asyncio
    async def test_on_trade_triggers_tfi_calculation(self, streaming_scanner, mock_calculator):
        """TFI is recalculated when enough trades are received."""
        streaming_scanner.add_symbol("TSLA")
        now = datetime.now(timezone.utc)

        for i in range(12):
            trade = StreamTrade(
                symbol="TSLA",
                price=200.0 + i,
                size=10,
                timestamp=now,
            )
            await streaming_scanner.on_trade("TSLA", trade)

        mock_calculator.calculate_tfi.assert_called()
        state = streaming_scanner._states["TSLA"]
        assert state.last_tfi == -0.3

    @pytest.mark.asyncio
    async def test_on_trade_tracks_session_high_low(self, streaming_scanner):
        """Session high/low tracking from trade prices."""
        streaming_scanner.add_symbol("SPY")
        now = datetime.now(timezone.utc)

        prices = [100.0, 105.0, 98.0, 102.0]
        for p in prices:
            trade = StreamTrade(symbol="SPY", price=p, size=10, timestamp=now)
            await streaming_scanner.on_trade("SPY", trade)

        state = streaming_scanner._states["SPY"]
        assert state.session_high == 105.0
        assert state.session_low == 98.0

    @pytest.mark.asyncio
    async def test_on_update_callback_fired(self, mock_calculator, mock_logger):
        """The on_update callback fires when feature delta exceeds threshold."""
        updates = []

        def capture_update(symbol, features):
            updates.append((symbol, features))

        scanner = StreamingScanner(
            calculator=mock_calculator,
            logger=mock_logger,
            on_update=capture_update,
        )
        scanner.add_symbol("AAPL")
        now = datetime.now(timezone.utc)

        # Push enough quotes for OFI calc to trigger
        for i in range(6):
            quote = StreamQuote(
                symbol="AAPL",
                bid_price=150.0 + i * 0.01,
                bid_size=100,
                ask_price=150.05,
                ask_size=200,
                timestamp=now,
            )
            await scanner.on_quote("AAPL", quote)

        assert len(updates) >= 1
        sym, feats = updates[0]
        assert sym == "AAPL"
        assert "ofi" in feats
        assert "tfi" in feats

    def test_reset_session(self, streaming_scanner):
        """reset_session clears all per-symbol state."""
        streaming_scanner.add_symbol("AAPL")
        state = streaming_scanner._states["AAPL"]
        state.last_ofi = 0.5
        state.last_tfi = -0.3
        state.session_high = 200.0
        state.session_low = 100.0

        streaming_scanner.reset_session()

        assert state.last_ofi == 0.0
        assert state.last_tfi == 0.0
        assert state.session_high == 0.0
        assert state.session_low == float("inf")
        assert len(state.recent_quotes) == 0
        assert len(state.recent_trades) == 0
