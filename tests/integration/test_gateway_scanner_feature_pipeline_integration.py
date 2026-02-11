"""Gateway-mode integration tests for scanner universe and feature pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import ConfigLoader
from src.core.settings import Settings
from src.data.pipeline import FeaturePipeline
from src.scanner.universe import UniverseBuilder


class _DummyConfigLoader(ConfigLoader):
    def __init__(self, config: dict):
        self._config = config

    def load_config(self, _path: str | None = None) -> dict:
        return dict(self._config)


def _gateway_runtime_settings() -> Settings:
    return Settings(
        CERBERUS_DATA_BACKEND="gateway",
        CERBERUS_FAILOVER_TO_LEGACY=False,
        CERBERUS_GATEWAY_URL="http://gateway.test",
        CERBERUS_GATEWAY_KEY="gw_test_key",
    )


@pytest.mark.integration
def test_gateway_mode_scanner_routes_screener_calls_to_central_api_client() -> None:
    runtime = _gateway_runtime_settings()
    logger = MagicMock()

    config = {
        "universe": {
            "symbols": [],
            "dynamic": {
                "screener": {
                    "enabled": True,
                    "most_actives_top_n": 2,
                    "movers_top_n": 1,
                }
            },
        }
    }

    central = MagicMock()
    central.get_alpaca_most_actives.return_value = ["AAPL", "MSFT"]
    central.get_alpaca_movers.return_value = {"gainers": ["TSLA"], "losers": ["NVDA"]}

    alpaca = MagicMock()
    alpaca.get_most_actives.side_effect = AssertionError("legacy screener should not be called in gateway mode")
    alpaca.get_movers.side_effect = AssertionError("legacy screener should not be called in gateway mode")

    with patch("src.scanner.universe.get_settings", return_value=runtime):
        builder = UniverseBuilder(
            _DummyConfigLoader(config),
            logger,
            config=config,
            alpaca_client=alpaca,
            central_api_client=central,
        )
        universe = builder.build_universe()

    assert universe == ["AAPL", "MSFT", "TSLA", "NVDA"]
    central.get_alpaca_most_actives.assert_called_once_with(top=2)
    central.get_alpaca_movers.assert_called_once_with(top=1)
    alpaca.get_most_actives.assert_not_called()
    alpaca.get_movers.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gateway_mode_feature_pipeline_uses_central_api_client_for_bars_and_trades() -> None:
    runtime = _gateway_runtime_settings()
    logger = MagicMock()

    bars_payload = {
        "bars": [
            {"t": "2025-01-06T14:30:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000},
            {"t": "2025-01-06T14:31:00Z", "o": 100.5, "h": 101.5, "l": 100.0, "c": 101.0, "v": 1100},
        ]
    }

    central = MagicMock()
    central.get_alpaca_bars.return_value = bars_payload
    central.get_alpaca_trades.return_value = [
        {"t": "2025-01-06T14:31:00Z", "p": 101.0, "s": 100, "c": ["@"], "x": "V", "z": "C"}
    ]

    alpaca = MagicMock()
    alpaca.get_historical_bars.side_effect = AssertionError("legacy bars should not be called in gateway mode")
    alpaca.get_historical_trades.side_effect = AssertionError("legacy trades should not be called in gateway mode")

    uw = AsyncMock()

    config = {
        "feature_pipeline": {"max_concurrency": 1},
        "unusual_whales": {"enabled": False},
    }

    with patch("src.data.fetcher.get_settings", return_value=runtime):
        pipeline = FeaturePipeline(
            alpaca_client=alpaca,
            unusual_whales_client=uw,
            logger=logger,
            config=config,
            central_api_client=central,
        )

    as_of = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)

    pipeline._fetch_supplementary_data = AsyncMock(return_value=(1_000_000.0, (101.0, 99.0, 100.0)))  # type: ignore[method-assign]
    pipeline.calculator.compute_technicals = MagicMock(
        return_value=SimpleNamespace(
            price=101.0,
            atr_pct=0.01,
            volume=1100.0,
            intraday_range_pct=0.02,
            gap_pct=0.0,
            ema20_slope=0.1,
            distance_from_vwap=0.001,
            premarket_volume=0.0,
            adx=25.0,
            distance_from_ema20=0.003,
            prior_day_high=101.0,
            prior_day_low=99.0,
            atr=1.2,
            bb_upper=102.0,
            bb_lower=98.0,
            price_zscore=0.2,
            tfi=0.0,
            timestamp=as_of,
            orb_high=101.5,
            orb_low=100.0,
            frac_diff_close=0.0,
            hurst_exponent=0.5,
        )
    )
    pipeline.calculator.calculate_relative_strength = MagicMock(return_value=0.1)
    pipeline.calculator.calculate_tfi = MagicMock(return_value=0.0)
    pipeline.calculator.apply_frac_diff = MagicMock(return_value=0.02)
    pipeline.calculator.calculate_hurst_exponent = MagicMock(return_value=0.55)
    pipeline.calculator.calculate_session_open_price = MagicMock(return_value=100.0)

    features = await pipeline.compute_technicals_only(["AAPL"], as_of=as_of)

    assert "AAPL" in features
    assert features["AAPL"].symbol == "AAPL"
    assert central.get_alpaca_bars.call_count >= 2  # SPY benchmark + symbol bars
    central.get_alpaca_trades.assert_called_once()
    alpaca.get_historical_bars.assert_not_called()
    alpaca.get_historical_trades.assert_not_called()
