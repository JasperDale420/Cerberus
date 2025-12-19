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
    # Return minute bars for intraday feature computation and daily bars for avg daily volume.
    mock_daily_bars = [{"v": 900000}, {"v": 1100000}, {"v": 1000000}]

    def _get_bars(_symbol, _start, _end, timeframe="1Min"):
        if timeframe == "1Day":
            return mock_daily_bars
        return mock_bars

    mock_alpaca.get_historical_bars.side_effect = _get_bars

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
    pipeline = FeaturePipeline(
        mock_alpaca,
        mock_uw,
        mock_logger,
        config={"feature_pipeline": {"daily_volume_lookback_days": 3}},
    )

    # Run
    features = await pipeline.compute_features([symbol], as_of=now)

    # Verify
    assert symbol in features
    feat = features[symbol]
    # Price should be last bar close (152 + 24 = 176)
    assert feat.price == 176.0
    assert feat.avg_volume == pytest.approx(1000000.0)
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

    features = await pipeline.compute_features(
        ["AAPL"], as_of=datetime.now(timezone.utc)
    )

    assert "AAPL" not in features
    mock_logger.warning.assert_called()


@pytest.mark.asyncio
async def test_compute_features_flow_failure_degrades_to_neutral():
    mock_alpaca = MagicMock()
    mock_uw = MagicMock()
    mock_logger = MagicMock()

    symbol = "AAPL"
    now = datetime.now(timezone.utc)

    # Provide enough bars for technicals.
    from datetime import timedelta

    mock_bars = []
    for i in range(25):
        t = now + timedelta(minutes=i)
        mock_bars.append(
            {
                "t": t.isoformat(),
                "o": 100.0 + i,
                "h": 101.0 + i,
                "l": 99.0 + i,
                "c": 100.5 + i,
                "v": 1000000,
            }
        )
    mock_daily_bars = [{"v": 1000000}] * 5

    def _get_bars(_symbol, _start, _end, timeframe="1Min"):
        if timeframe == "1Day":
            return mock_daily_bars
        return mock_bars

    mock_alpaca.get_historical_bars.side_effect = _get_bars

    # UW fails -> pipeline should return symbol with neutral flow features.
    mock_uw.get_option_flow = AsyncMock(side_effect=RuntimeError("uw down"))

    pipeline = FeaturePipeline(
        mock_alpaca,
        mock_uw,
        mock_logger,
        config={"feature_pipeline": {"daily_volume_lookback_days": 5}},
    )
    features = await pipeline.compute_features([symbol], as_of=now)

    assert symbol in features
    feat = features[symbol]
    assert feat.flow_zscore == 0.0
    assert feat.call_put_ratio == 0.0
    assert feat.large_sweeps_count == 0
    assert feat.aggressive_flow_share == 0.0
    mock_logger.warning.assert_called()
