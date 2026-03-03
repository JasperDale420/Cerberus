from __future__ import annotations

from src.main import _build_strategy_registry as build_runtime_strategy_registry


def test_runtime_strategy_registry_includes_configured_breakout_strategies() -> None:
    """Runtime registry should include strategies present in config defaults."""
    registry = build_runtime_strategy_registry()

    assert "trend_pullback" in registry
    assert "failed_breakout" in registry
