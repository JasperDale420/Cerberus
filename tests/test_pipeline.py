import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock unusualwhales module before importing src.data.pipeline
sys.modules["unusualwhales"] = MagicMock()

from datetime import datetime, timezone  # noqa: E402

from src.data.pipeline import FeaturePipeline  # noqa: E402


@pytest.mark.asyncio
async def test_compute_features():
    # Mock clients
    mock_alpaca = MagicMock()
    mock_uw = MagicMock()
    mock_logger = MagicMock()

    # Setup mock data
    symbol = "AAPL"
    now = datetime.now(timezone.utc)

    # Mock Alpaca response (Central API returns dict/list of dicts)
    # Need enough bars for technicals (EMA20, ATR14) - generate 25 bars
    mock_bars = []
    from datetime import timedelta

    for i in range(25):
        t = now + timedelta(minutes=i)
        mock_bars.append(
            {
                "t": t.isoformat(),
                "o": 150.0 + i,
                "h": 155.0 + i,
                "l": 149.0 + i,
                "c": 152.0 + i,
                "v": 1000000,
            }
        )
    mock_alpaca.get_historical_bars.return_value = mock_bars

    # Mock UW response (async) - List of trades
    mock_flow = [
        {
            "size": 10,
            "put_call": "CALL",
            "side": "BUY",
            "tags": [],
            "sentiment": "BULLISH",
        },
        {
            "size": 5,
            "put_call": "PUT",
            "side": "SELL",
            "tags": [],
            "sentiment": "BEARISH",
        },
    ]
    mock_uw.get_option_flow = AsyncMock(return_value=mock_flow)

    # Initialize pipeline
    pipeline = FeaturePipeline(mock_alpaca, mock_uw, mock_logger)

    # Run
    features = await pipeline.compute_features([symbol])

    # Verify
    assert symbol in features
    feat = features[symbol]
    # Price should be last bar close (152 + 24 = 176)
    assert feat.price == 176.0
    assert feat.avg_volume == 1000000
    assert feat.extra["flow_raw_count"] == 2

    # Verify calls
    # With data hardening, we may fetch prior day stats separately
    assert mock_alpaca.get_historical_bars.call_count >= 1
    mock_uw.get_option_flow.assert_called_once()


@pytest.mark.asyncio
async def test_compute_features_no_data():
    mock_alpaca = MagicMock()
    mock_uw = MagicMock()
    mock_logger = MagicMock()

    mock_alpaca.get_historical_bars.return_value = {}  # No data

    pipeline = FeaturePipeline(mock_alpaca, mock_uw, mock_logger)

    features = await pipeline.compute_features(["AAPL"])

    assert "AAPL" not in features
    mock_logger.warning.assert_called()
