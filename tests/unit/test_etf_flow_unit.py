"""Unit tests for the Cross-Asset ETF Flow Propagation analyzer."""

from __future__ import annotations

import pytest

from src.analysis.etf_flow import (
    DEFAULT_ETF_CONSTITUENTS,
    ETFFlowAnalyzer,
    ETFFlowConfig,
    ETFFlowResult,
    _FlowState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_etf_flow(symbol: str, call_premium: float, put_premium: float, net_premium: float) -> dict:
    return {
        "symbol": symbol,
        "call_premium": call_premium,
        "put_premium": put_premium,
        "net_premium": net_premium,
        "timestamp": "2026-03-12T10:00:00Z",
    }


def _make_stock_flow(symbol: str, call_premium: float, put_premium: float, net_premium: float) -> dict:
    return {
        "symbol": symbol,
        "call_premium": call_premium,
        "put_premium": put_premium,
        "net_premium": net_premium,
        "timestamp": "2026-03-12T10:00:00Z",
    }


def _seed_analyzer(
    analyzer: ETFFlowAnalyzer,
    etf_symbol: str = "SPY",
    stock_symbol: str = "AAPL",
    n: int = 25,
    etf_net: float = 100.0,
    stock_net: float = 50.0,
    etf_call: float = 200.0,
    etf_put: float = 100.0,
    stock_call: float = 100.0,
    stock_put: float = 50.0,
) -> None:
    """Feed *n* identical observations to fill rolling windows past min_bars."""
    for _ in range(n):
        analyzer.update_etf_flow(_make_etf_flow(etf_symbol, etf_call, etf_put, etf_net))
        analyzer.update_stock_flow(_make_stock_flow(stock_symbol, stock_call, stock_put, stock_net))


# ---------------------------------------------------------------------------
# Config defaults and overrides
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestETFFlowConfig:
    def test_defaults(self) -> None:
        cfg = ETFFlowConfig()
        assert cfg.enabled is False
        assert cfg.lookback_bars == 60
        assert cfg.min_bars == 20
        assert cfg.rotation_threshold == 1.5
        assert cfg.divergence_sigma == 2.0
        assert cfg.etf_symbols == ["SPY", "QQQ", "IWM"]

    def test_overrides(self) -> None:
        cfg = ETFFlowConfig(enabled=True, lookback_bars=30, min_bars=10, rotation_threshold=2.0)
        assert cfg.enabled is True
        assert cfg.lookback_bars == 30
        assert cfg.min_bars == 10
        assert cfg.rotation_threshold == 2.0


# ---------------------------------------------------------------------------
# Flow update and rolling window management
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlowUpdates:
    def test_etf_flow_update_populates_state(self) -> None:
        analyzer = ETFFlowAnalyzer()
        analyzer.update_etf_flow(_make_etf_flow("SPY", 200.0, 100.0, 100.0))
        state = analyzer._etf_states.get("SPY")
        assert state is not None
        assert state.count == 1
        assert state.net_premiums[-1] == 100.0
        # put/call = 100 / max(200, 1) = 0.5
        assert state.put_call_ratios[-1] == pytest.approx(0.5)

    def test_stock_flow_update_populates_state(self) -> None:
        analyzer = ETFFlowAnalyzer()
        analyzer.update_stock_flow(_make_stock_flow("AAPL", 300.0, 150.0, 150.0))
        state = analyzer._stock_states.get("AAPL")
        assert state is not None
        assert state.count == 1
        assert state.net_premiums[-1] == 150.0

    def test_rolling_window_respects_maxlen(self) -> None:
        cfg = ETFFlowConfig(lookback_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)
        for i in range(10):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 100.0, 50.0, float(i)))
        state = analyzer._etf_states["SPY"]
        assert state.count == 5  # maxlen enforced
        assert state.net_premiums[0] == 5.0  # oldest retained

    def test_empty_symbol_ignored(self) -> None:
        analyzer = ETFFlowAnalyzer()
        analyzer.update_etf_flow({"symbol": "", "call_premium": 1, "put_premium": 1, "net_premium": 1})
        assert len(analyzer._etf_states) == 0

    def test_case_insensitive_symbol(self) -> None:
        analyzer = ETFFlowAnalyzer()
        analyzer.update_etf_flow(_make_etf_flow("spy", 100.0, 50.0, 50.0))
        assert "SPY" in analyzer._etf_states

    def test_put_call_ratio_with_zero_calls(self) -> None:
        analyzer = ETFFlowAnalyzer()
        analyzer.update_etf_flow(_make_etf_flow("SPY", 0.0, 500.0, -500.0))
        state = analyzer._etf_states["SPY"]
        # put / max(call, 1.0) = 500 / 1.0 = 500.0
        assert state.put_call_ratios[-1] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Z-score computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestZScore:
    def test_zscore_insufficient_data(self) -> None:
        from collections import deque

        d: deque[float] = deque([1.0])
        assert ETFFlowAnalyzer._zscore(d) is None

    def test_zscore_zero_variance(self) -> None:
        from collections import deque

        d: deque[float] = deque([5.0, 5.0, 5.0, 5.0])
        assert ETFFlowAnalyzer._zscore(d) == pytest.approx(0.0)

    def test_zscore_known_value(self) -> None:
        from collections import deque

        # values: [0, 0, 0, 0, 10]
        # mean = 2, std = sqrt(mean of [4,4,4,4,64]) = sqrt(16) = 4
        # z = (10 - 2) / 4 = 2.0
        d: deque[float] = deque([0.0, 0.0, 0.0, 0.0, 10.0])
        z = ETFFlowAnalyzer._zscore(d)
        assert z is not None
        assert z == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Divergence score computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDivergence:
    def test_divergence_insufficient_data(self) -> None:
        analyzer = ETFFlowAnalyzer(config=ETFFlowConfig(min_bars=20))
        # Only feed 5 observations
        for _ in range(5):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 50, 50))
            analyzer.update_stock_flow(_make_stock_flow("AAPL", 100, 50, 50))
        result = analyzer.compute_divergence("AAPL", "SPY")
        assert result is None

    def test_divergence_missing_etf(self) -> None:
        analyzer = ETFFlowAnalyzer()
        _seed_analyzer(analyzer, stock_symbol="AAPL")
        result = analyzer.compute_divergence("AAPL", "QQQ")
        assert result is None

    def test_divergence_missing_stock(self) -> None:
        analyzer = ETFFlowAnalyzer()
        _seed_analyzer(analyzer, etf_symbol="SPY")
        result = analyzer.compute_divergence("TSLA", "SPY")
        assert result is None

    def test_divergence_stock_bullish_etf_bearish(self) -> None:
        """When stock flow spikes positive while ETF flow spikes negative,
        divergence should be significantly positive."""
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)

        # Feed baseline neutral flow
        for _ in range(20):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 100, 0.0))
            analyzer.update_stock_flow(_make_stock_flow("AAPL", 100, 100, 0.0))

        # ETF turns sharply bearish, stock turns sharply bullish
        for _ in range(5):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 50, 200, -150.0))
            analyzer.update_stock_flow(_make_stock_flow("AAPL", 200, 50, 150.0))

        div = analyzer.compute_divergence("AAPL", "SPY")
        assert div is not None
        assert div > 0  # stock bullish relative to ETF

    def test_divergence_equal_flow_near_zero(self) -> None:
        """When stock and ETF flow are identical, divergence should be ~0."""
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)

        for _ in range(25):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 100, 50.0))
            analyzer.update_stock_flow(_make_stock_flow("AAPL", 100, 100, 50.0))

        div = analyzer.compute_divergence("AAPL", "SPY")
        assert div is not None
        assert abs(div) < 0.01


# ---------------------------------------------------------------------------
# Rotation detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRotation:
    def test_rotation_spy_puts_surge_stock_calls_surge(self) -> None:
        analyzer = ETFFlowAnalyzer()
        # SPY put/call = 300 / max(100,1) = 3.0 > 1.5
        analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 300, -200))
        # AAPL put/call = 30 / max(200,1) = 0.15 < 0.5
        analyzer.update_stock_flow(_make_stock_flow("AAPL", 200, 30, 170))
        assert analyzer.detect_rotation("AAPL", "SPY") is True

    def test_no_rotation_when_etf_pcr_low(self) -> None:
        analyzer = ETFFlowAnalyzer()
        # SPY put/call = 50 / 200 = 0.25 < 1.5
        analyzer.update_etf_flow(_make_etf_flow("SPY", 200, 50, 150))
        analyzer.update_stock_flow(_make_stock_flow("AAPL", 200, 30, 170))
        assert analyzer.detect_rotation("AAPL", "SPY") is False

    def test_no_rotation_when_stock_pcr_high(self) -> None:
        analyzer = ETFFlowAnalyzer()
        # SPY put surge
        analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 300, -200))
        # AAPL put/call = 200 / 100 = 2.0 > 0.5
        analyzer.update_stock_flow(_make_stock_flow("AAPL", 100, 200, -100))
        assert analyzer.detect_rotation("AAPL", "SPY") is False

    def test_rotation_missing_data(self) -> None:
        analyzer = ETFFlowAnalyzer()
        assert analyzer.detect_rotation("AAPL", "SPY") is False

    def test_rotation_custom_threshold(self) -> None:
        cfg = ETFFlowConfig(rotation_threshold=3.0)
        analyzer = ETFFlowAnalyzer(config=cfg)
        # SPY put/call = 2.0 < 3.0 threshold
        analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 200, -100))
        analyzer.update_stock_flow(_make_stock_flow("AAPL", 200, 30, 170))
        assert analyzer.detect_rotation("AAPL", "SPY") is False


# ---------------------------------------------------------------------------
# Lead signal
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeadSignal:
    def test_lead_signal_insufficient_data(self) -> None:
        analyzer = ETFFlowAnalyzer(config=ETFFlowConfig(min_bars=20))
        for _ in range(5):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 100, 50))
        signal, conf, lag = analyzer.compute_lead_signal("SPY")
        assert signal == 0.0
        assert conf == 0.0
        assert lag == 10

    def test_lead_signal_sharp_bearish_reversal(self) -> None:
        """When ETF flow reverses sharply negative, lead signal should be negative."""
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)

        # Gradually increasing positive flow to build a baseline of small positive changes
        for i in range(20):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 200, 100, 100.0 + i * 2.0))

        # Sharp negative reversal: a large negative change from the last value
        last_positive = 100.0 + 19 * 2.0  # 138.0
        analyzer.update_etf_flow(_make_etf_flow("SPY", 50, 300, last_positive - 500.0))

        signal, conf, lag = analyzer.compute_lead_signal("SPY")
        assert signal < 0  # bearish lead
        assert conf > 0
        assert lag == 10

    def test_lead_signal_sharp_bullish_reversal(self) -> None:
        """When ETF flow reverses sharply positive, lead signal should be positive."""
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)

        # Gradually decreasing negative flow to build a baseline of small negative changes
        for i in range(20):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 50, 200, -150.0 - i * 2.0))

        # Sharp positive reversal: a large positive change from the last value
        last_negative = -150.0 - 19 * 2.0  # -188.0
        analyzer.update_etf_flow(_make_etf_flow("SPY", 300, 50, last_negative + 500.0))

        signal, conf, lag = analyzer.compute_lead_signal("SPY")
        assert signal > 0  # bullish lead
        assert conf > 0

    def test_lead_signal_clamped_to_range(self) -> None:
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)

        # Flat baseline
        for _ in range(25):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 100, 0.0))

        # Extreme spike
        analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 100, 999999.0))

        signal, conf, lag = analyzer.compute_lead_signal("SPY")
        assert -1.0 <= signal <= 1.0
        assert 0.0 <= conf <= 1.0

    def test_lead_signal_zero_variance(self) -> None:
        """Constant flow changes should yield zero signal."""
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)

        for _ in range(25):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 100, 50.0))

        signal, conf, lag = analyzer.compute_lead_signal("SPY")
        assert signal == pytest.approx(0.0)
        assert conf == pytest.approx(0.0)

    def test_lead_signal_unknown_etf(self) -> None:
        analyzer = ETFFlowAnalyzer()
        signal, conf, lag = analyzer.compute_lead_signal("XYZ")
        assert signal == 0.0
        assert conf == 0.0


# ---------------------------------------------------------------------------
# Composite analyze
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyze:
    def test_analyze_returns_result(self) -> None:
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)
        _seed_analyzer(analyzer, n=25)

        result = analyzer.analyze("AAPL", "SPY")
        assert result is not None
        assert isinstance(result, ETFFlowResult)
        assert result.etf_symbol == "SPY"
        assert result.stock_symbol == "AAPL"
        assert isinstance(result.divergence_score, float)
        assert isinstance(result.rotation_detected, bool)
        assert -1.0 <= result.lead_signal <= 1.0

    def test_analyze_insufficient_data_returns_none(self) -> None:
        analyzer = ETFFlowAnalyzer(config=ETFFlowConfig(min_bars=20))
        for _ in range(5):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 100, 100, 50))
            analyzer.update_stock_flow(_make_stock_flow("AAPL", 100, 100, 50))
        assert analyzer.analyze("AAPL", "SPY") is None

    def test_analyze_missing_stock_returns_none(self) -> None:
        analyzer = ETFFlowAnalyzer()
        _seed_analyzer(analyzer, etf_symbol="SPY")
        assert analyzer.analyze("MISSING", "SPY") is None

    def test_analyze_frozen_result(self) -> None:
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)
        _seed_analyzer(analyzer, n=25)

        result = analyzer.analyze("AAPL", "SPY")
        assert result is not None
        with pytest.raises(AttributeError):
            result.divergence_score = 999.0  # type: ignore[misc]

    def test_analyze_includes_put_call_ratios(self) -> None:
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)
        _seed_analyzer(
            analyzer,
            n=25,
            etf_call=200.0,
            etf_put=100.0,
            stock_call=100.0,
            stock_put=50.0,
        )

        result = analyzer.analyze("AAPL", "SPY")
        assert result is not None
        assert result.etf_put_call_ratio == pytest.approx(0.5)  # 100/200
        assert result.stock_put_call_ratio == pytest.approx(0.5)  # 50/100


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    def test_zero_flow_all_around(self) -> None:
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)

        for _ in range(25):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 0.0, 0.0, 0.0))
            analyzer.update_stock_flow(_make_stock_flow("AAPL", 0.0, 0.0, 0.0))

        div = analyzer.compute_divergence("AAPL", "SPY")
        assert div is not None
        assert div == pytest.approx(0.0)

        assert analyzer.detect_rotation("AAPL", "SPY") is False

    def test_multiple_etfs_independent(self) -> None:
        cfg = ETFFlowConfig(lookback_bars=30, min_bars=5)
        analyzer = ETFFlowAnalyzer(config=cfg)

        # Use varying values so z-scores are non-zero and differ between ETFs
        for i in range(20):
            analyzer.update_etf_flow(_make_etf_flow("SPY", 200, 100, 50.0 + i))
            analyzer.update_etf_flow(_make_etf_flow("QQQ", 100, 200, -50.0 - i))
            analyzer.update_stock_flow(_make_stock_flow("AAPL", 150, 75, 40.0 + i))

        # Final spike: stock diverges in opposite direction from QQQ vs SPY
        analyzer.update_etf_flow(_make_etf_flow("SPY", 200, 100, 200.0))
        analyzer.update_etf_flow(_make_etf_flow("QQQ", 100, 200, -200.0))
        analyzer.update_stock_flow(_make_stock_flow("AAPL", 150, 75, 200.0))

        spy_div = analyzer.compute_divergence("AAPL", "SPY")
        qqq_div = analyzer.compute_divergence("AAPL", "QQQ")
        assert spy_div is not None
        assert qqq_div is not None
        # They should differ because ETF flows are in opposite directions
        assert spy_div != qqq_div

    def test_flow_state_internal_maxlen(self) -> None:
        state = _FlowState(maxlen=3)
        for i in range(10):
            state.update(float(i), float(i) * 0.1)
        assert state.count == 3
        assert list(state.net_premiums) == [7.0, 8.0, 9.0]

    def test_default_etf_constituents_structure(self) -> None:
        assert "SPY" in DEFAULT_ETF_CONSTITUENTS
        assert "QQQ" in DEFAULT_ETF_CONSTITUENTS
        assert "IWM" in DEFAULT_ETF_CONSTITUENTS
        # All have empty lists (loose mapping)
        for v in DEFAULT_ETF_CONSTITUENTS.values():
            assert isinstance(v, list)
