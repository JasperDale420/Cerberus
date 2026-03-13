from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.domain import SymbolFeatures
from src.scanner.core import Scanner


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
    """Scanner assigns strategies from strategy_routing config by regime."""
    mock_universe = MagicMock()
    mock_universe.get_universe.return_value = ["AAPL", "TSLA"]

    mock_pipeline = MagicMock()
    mock_features = {
        "AAPL": create_features("AAPL", price=150.0, volume=200000),
        "TSLA": create_features("TSLA", price=200.0, volume=50000),
    }
    mock_pipeline.compute_technicals_only = AsyncMock(return_value=mock_features)
    mock_pipeline.append_flow_features = AsyncMock(return_value=mock_features)

    mock_logger = MagicMock()

    config = {
        "strategy_routing": {
            "chop": ["vwap_reversion", "failed_breakout", "index_mean_reversion", "gap_fill"],
        },
    }

    scanner = Scanner(mock_universe, mock_pipeline, mock_logger, config=config)
    results = await scanner.scan(scan_time=datetime.now(timezone.utc))

    # Both symbols should be in the watchlist (no profile filtering)
    matched_symbols = [s.symbol for s in results.watchlist]
    assert "AAPL" in matched_symbols
    assert "TSLA" in matched_symbols

    # All symbols get the regime's strategies
    aapl_res = next(s for s in results.watchlist if s.symbol == "AAPL")
    assert "vwap_reversion" in aapl_res.strategies
    assert "failed_breakout" in aapl_res.strategies
    assert "index_mean_reversion" in aapl_res.strategies
    assert "gap_fill" in aapl_res.strategies

    # Universe and pipeline called correctly
    mock_universe.get_universe.assert_called_once()
    assert mock_pipeline.compute_technicals_only.call_count == 1
    args, kwargs = mock_pipeline.compute_technicals_only.call_args
    assert args[0] == ["AAPL", "TSLA"]
    assert kwargs.get("as_of") is not None
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
@pytest.mark.asyncio
async def test_scanner_empty_routing_produces_empty_strategies():
    """When strategy_routing has no entry for the regime, watchlist symbols get no strategies."""
    mock_universe = MagicMock()
    mock_universe.get_universe.return_value = ["AAPL"]

    mock_pipeline = MagicMock()
    mock_features = {"AAPL": create_features("AAPL", price=150.0, volume=200000)}
    mock_pipeline.compute_technicals_only = AsyncMock(return_value=mock_features)
    mock_pipeline.append_flow_features = AsyncMock(return_value=mock_features)

    scanner = Scanner(mock_universe, mock_pipeline, MagicMock(), config={})
    results = await scanner.scan(scan_time=datetime.now(timezone.utc))

    assert len(results.watchlist) == 1
    assert results.watchlist[0].symbol == "AAPL"
    assert results.watchlist[0].strategies == []
