import pytest

from src.data.calculator import FeatureCalculator


@pytest.fixture
def calculator():
    return FeatureCalculator()


def test_compute_technicals_empty(calculator):
    assert calculator.compute_technicals([]) is None


def test_compute_technicals_basic(calculator):
    # minimal bar structure mock
    bars = [
        {"t": "2025-01-01T09:30:00Z", "o": 100, "h": 105, "l": 95, "c": 102, "v": 1000},
        {"t": "2025-01-01T09:31:00Z", "o": 102, "h": 103, "l": 101, "c": 101, "v": 500},
        {"t": "2025-01-01T09:32:00Z", "o": 101, "h": 104, "l": 100, "c": 103, "v": 800},
    ]
    result = calculator.compute_technicals(bars)
    assert result is not None

    # TechnicalFeatures is a dataclass - access attributes directly
    assert result.price == 103.0
    assert result.volume == 800.0
    assert result.timestamp.isoformat() == "2025-01-01T09:32:00+00:00"
    assert result.atr_pct > 0


def test_compute_flow_metrics_empty(calculator):
    metrics = calculator.compute_flow_metrics([])
    assert metrics == (0.0, 0.0, 0, 0.0, 0.0)


def test_compute_flow_metrics_bullish(calculator):
    flow = [
        {"size": 100, "put_call": "CALL", "tags": ["sweep"], "sentiment": "BULLISH"},
        {"size": 50, "put_call": "PUT", "tags": [], "sentiment": "BEARISH"},
    ]
    (cp_ratio, zscore, sweeps, agg_share, bias) = calculator.compute_flow_metrics(flow)

    assert cp_ratio == 2.0
    assert sweeps == 1
    assert bias > 0  # Call heavy
