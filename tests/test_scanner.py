import pytest
from unittest.mock import MagicMock, AsyncMock
from src.scanner.core import Scanner
from src.scanner.universe import UniverseBuilder
from src.scanner.profiles import VWAPReversionProfile
from src.data.models import SymbolFeatures
from datetime import datetime

@pytest.mark.asyncio
async def test_scanner_flow():
    # Mocks
    mock_universe = MagicMock()
    mock_universe.get_universe.return_value = ["AAPL", "TSLA"]
    
    mock_pipeline = MagicMock()
    # Mock features return
    mock_features = {
        "AAPL": SymbolFeatures(
            symbol="AAPL", timestamp=datetime.utcnow(), price=150.0, volume=200000, 
            flow_sentiment=0, volatility=0, extra={}
        ),
        "TSLA": SymbolFeatures(
            symbol="TSLA", timestamp=datetime.utcnow(), price=200.0, volume=50000, # Low volume
            flow_sentiment=0, volatility=0, extra={}
        )
    }
    mock_pipeline.compute_features = AsyncMock(return_value=mock_features)
    
    mock_logger = MagicMock()
    
    # Init Scanner
    scanner = Scanner(mock_universe, mock_pipeline, mock_logger)
    
    # Run scan
    results = await scanner.scan()
    
    # Verify
    # AAPL should pass VWAP profile (default min_vol=100k)
    # TSLA should fail (vol=50k < 100k)
    
    assert len(results) == 1
    res = results[0]
    assert res.symbol == "AAPL"
    assert "vwap_reversion" in res.matching_strategies
    
    # Verify calls
    mock_universe.get_universe.assert_called_once()
    mock_pipeline.compute_features.assert_called_once_with(["AAPL", "TSLA"])

def test_vwap_profile():
    profile = VWAPReversionProfile(min_price=10.0, min_volume=1000)
    
    # Pass
    f1 = SymbolFeatures("A", datetime.utcnow(), 15.0, 2000, 0, 0, {})
    assert profile.filter(f1) is True
    
    # Fail Price
    f2 = SymbolFeatures("B", datetime.utcnow(), 5.0, 2000, 0, 0, {})
    assert profile.filter(f2) is False
    
    # Fail Volume
    f3 = SymbolFeatures("C", datetime.utcnow(), 15.0, 500, 0, 0, {})
    assert profile.filter(f3) is False
