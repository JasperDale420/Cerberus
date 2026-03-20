import pandas as pd
import pytest


@pytest.mark.unit
def test_detect_gaps_in_bars():
    from src.backtest.data_quality import check_data_quality

    timestamps = pd.date_range("2024-01-02 09:30", periods=100, freq="1min", tz="US/Eastern")
    timestamps = timestamps.delete(range(50, 55))
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 150.0,
            "high": 151.0,
            "low": 149.0,
            "close": 150.5,
            "volume": 1000,
            "symbol": "AAPL",
        }
    )
    report = check_data_quality({"AAPL": df})
    assert report.symbols["AAPL"].gap_count >= 1
    assert report.symbols["AAPL"].coverage_pct < 100.0


@pytest.mark.unit
def test_detect_zero_volume():
    from src.backtest.data_quality import check_data_quality

    timestamps = pd.date_range("2024-01-02 09:30", periods=50, freq="1min", tz="US/Eastern")
    volumes = [1000] * 50
    volumes[10] = 0
    volumes[20] = 0
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 150.0,
            "high": 151.0,
            "low": 149.0,
            "close": 150.5,
            "volume": volumes,
            "symbol": "AAPL",
        }
    )
    report = check_data_quality({"AAPL": df})
    assert report.symbols["AAPL"].zero_volume_bars == 2


@pytest.mark.unit
def test_detect_price_outliers():
    from src.backtest.data_quality import check_data_quality

    timestamps = pd.date_range("2024-01-02 09:30", periods=50, freq="1min", tz="US/Eastern")
    closes = [150.0] * 50
    closes[25] = 200.0
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 150.0,
            "high": 151.0,
            "low": 149.0,
            "close": closes,
            "volume": 1000,
            "symbol": "AAPL",
        }
    )
    report = check_data_quality({"AAPL": df})
    assert report.symbols["AAPL"].outlier_count >= 1


@pytest.mark.unit
def test_exclude_low_coverage_symbol():
    from src.backtest.data_quality import check_data_quality

    timestamps = pd.date_range("2024-01-02 09:30", periods=10, freq="1min", tz="US/Eastern")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 150.0,
            "high": 151.0,
            "low": 149.0,
            "close": 150.5,
            "volume": 1000,
            "symbol": "BAD",
        }
    )
    report = check_data_quality({"BAD": df}, min_coverage_pct=80.0, exclude_below_pct=50.0)
    assert "BAD" in report.excluded_symbols


@pytest.mark.unit
def test_detect_stale_prices():
    from src.backtest.data_quality import check_data_quality

    timestamps = pd.date_range("2024-01-02 09:30", periods=50, freq="1min", tz="US/Eastern")
    closes = [150.0] * 50
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 150.0,
            "high": 151.0,
            "low": 149.0,
            "close": closes,
            "volume": 1000,
            "symbol": "STALE",
        }
    )
    report = check_data_quality({"STALE": df})
    assert report.symbols["STALE"].stale_streak > 10
