from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.domain import SymbolFeatures
from src.scanner.core import Scanner
from src.scanner.profiles import VWAPReversionProfile


def create_features(symbol, price=100.0, volume=100000.0):
    return SymbolFeatures(
        symbol=symbol,
        price=price,
        atr_pct=1.0,
        avg_volume=volume,
        intraday_range_pct=2.0,
        gap_pct=0.5,
        ema20_slope=0.1,
        ema_trend_strength=0.5,
        distance_from_vwap=0.0,
        premarket_volume=1000.0,
        adx=25.0,
        distance_from_ema20=0.0,
        prior_day_high=price * 1.01,
        prior_day_low=price * 0.99,
        bb_upper=price * 1.02,
        bb_lower=price * 0.98,
        price_zscore=0.5,
        flow_zscore=0.0,
        call_put_ratio=0.5,
        large_sweeps_count=0,
        aggressive_flow_share=0.0,
        last_updated=datetime.now(timezone.utc),
        extra={},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scanner_flow():
    # Mocks
    mock_universe = MagicMock()
    mock_universe.get_universe.return_value = ["AAPL", "TSLA"]

    mock_pipeline = MagicMock()
    # Mock features return (Stage 1 technicals)
    mock_features = {
        "AAPL": create_features("AAPL", price=150.0, volume=200000),
        "TSLA": create_features("TSLA", price=200.0, volume=50000),  # Low volume
    }
    mock_pipeline.compute_technicals_only = AsyncMock(return_value=mock_features)
    mock_pipeline.append_flow_features = AsyncMock(return_value=mock_features)

    mock_logger = MagicMock()

    # Init Scanner
    scanner = Scanner(mock_universe, mock_pipeline, mock_logger)

    # Run scan
    results = await scanner.scan(scan_time=datetime.now(timezone.utc))

    # Verify
    # TSLA might match other profiles added recently (e.g. GapFill), so allow len >= 1
    # Check that AAPL is matched and has VWAP strategy
    matched_symbols = [s.symbol for s in results.watchlist]
    assert "AAPL" in matched_symbols

    aapl_res = next(s for s in results.watchlist if s.symbol == "AAPL")
    assert "vwap_reversion" in aapl_res.strategies

    # Verify calls
    mock_universe.get_universe.assert_called_once()
    # Determinism: scan_time is passed through as_of
    assert mock_pipeline.compute_technicals_only.call_count == 1
    args, kwargs = mock_pipeline.compute_technicals_only.call_args
    assert args[0] == ["AAPL", "TSLA"]
    assert kwargs.get("as_of") is not None

    # Verify Stage 2 called for survivors
    # AAPL passed valid price/vol filters; TSLA also passed?
    # min_volume default 0.0, so both pass baseline.
    assert mock_pipeline.append_flow_features.call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scanner_requires_scan_time_or_pipeline_clock() -> None:
    mock_universe = MagicMock()
    mock_universe.get_universe.return_value = ["AAPL"]

    mock_pipeline = MagicMock(spec=["compute_technicals_only"])
    mock_pipeline.compute_technicals_only = AsyncMock(return_value={})

    scanner = Scanner(mock_universe, mock_pipeline, MagicMock())

    with pytest.raises(ValueError, match="requires scan_time"):
        await scanner.scan()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scanner_feature_pipeline_failure_fails_open() -> None:
    mock_universe = MagicMock()
    mock_universe.get_universe.return_value = ["AAPL"]

    mock_pipeline = MagicMock()
    mock_pipeline.compute_technicals_only = AsyncMock(side_effect=RuntimeError("boom"))

    mock_logger = MagicMock()
    scanner = Scanner(mock_universe, mock_pipeline, mock_logger)

    result = await scanner.scan(scan_time=datetime.now(timezone.utc))
    assert result.watchlist == []
    assert mock_logger.error.call_count >= 1


@pytest.mark.unit
def test_vwap_profile():
    profile = VWAPReversionProfile(min_price=10.0, min_volume=1000)

    # Pass
    f1 = create_features("A", price=15.0, volume=2000)
    assert profile.filter(f1) is True

    # Fail Price
    f2 = create_features("B", price=5.0, volume=2000)
    assert profile.filter(f2) is False

    # Fail Volume
    f3 = create_features("C", price=15.0, volume=500)
    assert profile.filter(f3) is False


@pytest.mark.unit
def test_scanner_validation_defaults_on_bad_config_values():
    mock_universe = MagicMock()
    mock_pipeline = MagicMock()
    mock_logger = MagicMock()
    config = {
        "scanner": {
            "min_price": "nope",
            "max_price": None,
            "min_volume": "bad",
            "min_atr_pct": "n/a",
            "max_atr_pct": "n/a",
        }
    }
    scanner = Scanner(mock_universe, mock_pipeline, mock_logger, config=config)
    features_map = {"AAPL": create_features("AAPL", price=100.0, volume=100000.0)}

    survivors, filtered = scanner._apply_data_validation(features_map)

    assert "AAPL" in survivors
    assert filtered == 0
    assert mock_logger.warning.call_count >= 1
