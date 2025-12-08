import sys
import pytest
from unittest.mock import MagicMock, AsyncMock

# Mock unusualwhales module before importing src.data.pipeline
sys.modules["unusualwhales"] = MagicMock()

from datetime import datetime
from src.data.pipeline import FeaturePipeline
from src.data.models import Bar

@pytest.mark.asyncio
async def test_compute_features():
    # Mock clients
    mock_alpaca = MagicMock()
    mock_uw = MagicMock()
    mock_logger = MagicMock()
    
    # Setup mock data
    symbol = "AAPL"
    now = datetime.utcnow()
    
    # Mock Alpaca response (Central API returns dict/list of dicts)
    mock_bar_data = {
        "t": now.isoformat(),
        "o": 150.0,
        "h": 155.0,
        "l": 149.0,
        "c": 152.0,
        "v": 1000000
    }
    mock_alpaca.get_historical_bars.return_value = [mock_bar_data]
    
    # Mock UW response (async)
    mock_uw.get_option_flow = AsyncMock(return_value={"some": "data"})
    
    # Initialize pipeline
    pipeline = FeaturePipeline(mock_alpaca, mock_uw, mock_logger)
    
    # Run
    features = await pipeline.compute_features([symbol])
    
    # Verify
    assert symbol in features
    feat = features[symbol]
    assert feat.price == 152.0
    assert feat.volume == 1000000
    assert feat.extra["flow_raw"] == {"some": "data"}
    
    # Verify calls
    mock_alpaca.get_historical_bars.assert_called_once()
    mock_uw.get_option_flow.assert_called_once()

@pytest.mark.asyncio
async def test_compute_features_no_data():
    mock_alpaca = MagicMock()
    mock_uw = MagicMock()
    mock_logger = MagicMock()
    
    mock_alpaca.get_historical_bars.return_value = {} # No data
    
    pipeline = FeaturePipeline(mock_alpaca, mock_uw, mock_logger)
    
    features = await pipeline.compute_features(["AAPL"])
    
    assert "AAPL" not in features
    mock_logger.warning.assert_called()
