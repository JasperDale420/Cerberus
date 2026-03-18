"""Tests for HMM regime gate in BaseStrategy."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.domain import MarketState
from src.strategies.base import BaseStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockLogger:
    """Minimal mock matching StructuredLogger interface."""

    def __init__(self):
        self.messages: list[tuple[str, str, dict]] = []

    def debug(self, msg: str, **kw: Any) -> None:
        self.messages.append(("debug", msg, kw))

    def info(self, msg: str, **kw: Any) -> None:
        self.messages.append(("info", msg, kw))

    def warning(self, msg: str, **kw: Any) -> None:
        self.messages.append(("warning", msg, kw))

    def error(self, msg: str, **kw: Any) -> None:
        self.messages.append(("error", msg, kw))

    def critical(self, msg: str, **kw: Any) -> None:
        self.messages.append(("critical", msg, kw))


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""

    name = "test_strategy"

    def on_bar(self, symbol, bar, symbol_state, market_state):
        return None


def _make_market_state(hmm_regime: dict | None = None) -> MarketState:
    ms = MagicMock(spec=MarketState)
    ms.meta = {}
    if hmm_regime is not None:
        ms.meta["hmm_regime"] = hmm_regime
    return ms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHMMGate:
    """Test suite for BaseStrategy._check_hmm_gate()."""

    def test_passes_when_not_configured(self):
        """HMM gate passes when enabled=false (default)."""
        strategy = ConcreteStrategy({"cooldown_bars": 0}, _MockLogger())
        ms = _make_market_state({"label": "stress_up", "confidence": 0.9})
        assert strategy._check_hmm_gate(ms) is True

    def test_passes_when_no_hmm_prediction(self):
        """HMM gate passes when no HMM prediction in market_state.meta."""
        config = {"hmm_gate": {"enabled": True, "reject_regimes": ["stress_up"], "min_confidence": 0.6}}
        strategy = ConcreteStrategy(config, _MockLogger())
        ms = _make_market_state(hmm_regime=None)
        assert strategy._check_hmm_gate(ms) is True

    def test_passes_when_empty_hmm_dict(self):
        """HMM gate passes when hmm_regime is an empty dict."""
        config = {"hmm_gate": {"enabled": True, "reject_regimes": ["stress_up"], "min_confidence": 0.6}}
        strategy = ConcreteStrategy(config, _MockLogger())
        ms = _make_market_state(hmm_regime={})
        assert strategy._check_hmm_gate(ms) is True

    def test_passes_when_confidence_below_threshold(self):
        """HMM gate passes when confidence is below min_confidence."""
        config = {"hmm_gate": {"enabled": True, "reject_regimes": ["stress_up"], "min_confidence": 0.6}}
        strategy = ConcreteStrategy(config, _MockLogger())
        ms = _make_market_state({"label": "stress_up", "confidence": 0.4})
        assert strategy._check_hmm_gate(ms) is True

    def test_rejects_when_label_in_reject_regimes(self):
        """HMM gate rejects when label matches reject_regimes and confidence >= min."""
        config = {"hmm_gate": {"enabled": True, "reject_regimes": ["stress_up"], "min_confidence": 0.6}}
        logger = _MockLogger()
        strategy = ConcreteStrategy(config, logger)
        ms = _make_market_state({"label": "stress_up", "confidence": 0.8})
        assert strategy._check_hmm_gate(ms) is False
        # Verify debug log was emitted
        assert any(msg == "hmm_gate_rejected" for _, msg, _ in logger.messages)

    def test_passes_when_label_not_in_reject_regimes(self):
        """HMM gate passes when label is not in reject_regimes."""
        config = {"hmm_gate": {"enabled": True, "reject_regimes": ["stress_up"], "min_confidence": 0.6}}
        strategy = ConcreteStrategy(config, _MockLogger())
        ms = _make_market_state({"label": "trend_up", "confidence": 0.9})
        assert strategy._check_hmm_gate(ms) is True

    def test_rejects_multiple_reject_regimes(self):
        """HMM gate rejects any label listed in reject_regimes."""
        config = {"hmm_gate": {"enabled": True, "reject_regimes": ["trend_up", "trend_down"], "min_confidence": 0.6}}
        strategy = ConcreteStrategy(config, _MockLogger())

        ms_up = _make_market_state({"label": "trend_up", "confidence": 0.8})
        assert strategy._check_hmm_gate(ms_up) is False

        ms_down = _make_market_state({"label": "trend_down", "confidence": 0.8})
        assert strategy._check_hmm_gate(ms_down) is False

        ms_neutral = _make_market_state({"label": "neutral", "confidence": 0.8})
        assert strategy._check_hmm_gate(ms_neutral) is True

    def test_hmm_gate_enabled_property_set(self):
        """_hmm_gate_enabled property is set from config."""
        config_enabled = {"hmm_gate": {"enabled": True, "reject_regimes": ["stress_up"]}}
        strategy_on = ConcreteStrategy(config_enabled, _MockLogger())
        assert strategy_on._hmm_gate_enabled is True

        config_disabled = {"hmm_gate": {"enabled": False}}
        strategy_off = ConcreteStrategy(config_disabled, _MockLogger())
        assert strategy_off._hmm_gate_enabled is False

        strategy_default = ConcreteStrategy({}, _MockLogger())
        assert strategy_default._hmm_gate_enabled is False

    def test_default_min_confidence_is_half(self):
        """When min_confidence is not specified, defaults to 0.5."""
        config = {"hmm_gate": {"enabled": True, "reject_regimes": ["stress_up"]}}
        strategy = ConcreteStrategy(config, _MockLogger())

        # Confidence 0.4 < default 0.5 -> should pass (not confident enough to gate)
        ms_low = _make_market_state({"label": "stress_up", "confidence": 0.4})
        assert strategy._check_hmm_gate(ms_low) is True

        # Confidence 0.6 >= default 0.5 -> should reject
        ms_high = _make_market_state({"label": "stress_up", "confidence": 0.6})
        assert strategy._check_hmm_gate(ms_high) is False
