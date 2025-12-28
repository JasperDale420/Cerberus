from unittest.mock import patch

import pytest

from src.backtest.runner import BacktestRunner


@pytest.fixture
def mock_deps():
    with (
        patch("src.backtest.runner.ConfigLoader") as mock_conf_loader,
        patch("src.backtest.runner.AlpacaClient") as mock_alpaca,
        patch("src.backtest.runner.UniverseBuilder") as mock_universe_builder,
        patch("src.backtest.runner.ExecutionEngine") as mock_engine,
        patch("src.backtest.runner.BacktestOrderExecutor") as mock_executor,
    ):

        # Setup Config
        mock_conf_instance = mock_conf_loader.return_value
        mock_conf_instance.load_config.return_value = {"timeframe": "1Min"}

        # Setup Universe
        mock_ub_instance = mock_universe_builder.return_value
        mock_ub_instance.build_universe.return_value = ["AAPL", "GOOG"]

        yield {
            "config_loader": mock_conf_loader,
            "alpaca": mock_alpaca,
            "universe_builder": mock_universe_builder,
            "engine": mock_engine,
            "executor": mock_executor,
        }


@pytest.mark.asyncio
async def test_backtest_runner_uses_universe_builder(mock_deps):
    """
    Verify that BacktestRunner initializes UniverseBuilder and uses the returned universe.
    """
    runner = BacktestRunner(
        config_path="dummy_config.yaml",
        start_date="2025-01-01T09:30:00",
        end_date="2025-01-01T16:00:00",
    )

    # Check UniverseBuilder instantiation
    mock_deps["universe_builder"].assert_called_once()

    # Check universe population
    assert runner.universe == ["AAPL", "GOOG"]

    # Mock get_historical_bars to return empty list to avoid iteration errors
    runner.alpaca_client.get_historical_bars.return_value = []  # type: ignore

    # Run
    await runner.run()

    # Verify data fetching was attempted for both symbols
    assert runner.alpaca_client.get_historical_bars.call_count == 2  # type: ignore
    calls = runner.alpaca_client.get_historical_bars.call_args_list  # type: ignore
    symbols_fetched = {c[0][0] for c in calls}  # Arg 0 is symbol
    assert "AAPL" in symbols_fetched
    assert "GOOG" in symbols_fetched
