from src.data.requirements import DataRequirements
from src.strategies.base import BaseStrategy


def test_base_strategy_has_default_requirements():
    assert hasattr(BaseStrategy, "data_requirements")
    assert isinstance(BaseStrategy.data_requirements, DataRequirements)


def test_flow_alpha_needs_flow():
    from src.strategies.flow_alpha import FlowAlphaStrategy

    assert "flow" in FlowAlphaStrategy.data_requirements.on_scan


def test_order_flow_imbalance_needs_trades():
    from src.strategies.order_flow_imbalance import OrderFlowImbalanceStrategy

    assert "trades" in OrderFlowImbalanceStrategy.data_requirements.streams


def test_all_registered_strategies_have_requirements():
    from src.main import _build_strategy_registry

    registry = _build_strategy_registry()
    for name, cls in registry.items():
        assert hasattr(cls, "data_requirements"), f"{name} missing data_requirements"
        assert isinstance(cls.data_requirements, DataRequirements), f"{name} has wrong type"
