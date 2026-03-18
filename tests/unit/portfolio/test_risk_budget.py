"""Tests for portfolio risk budget with VaR/CVaR and concentration limits."""

import pytest

from src.portfolio.risk_budget import PortfolioRiskBudget, RiskBudgetResult


@pytest.fixture
def budget() -> PortfolioRiskBudget:
    return PortfolioRiskBudget(
        max_portfolio_cvar=500.0,
        max_marginal_cvar_pct=0.25,
        max_strategy_risk_pct=0.40,
        max_symbol_risk_pct=0.25,
    )


class TestEmptyPortfolio:
    """Empty portfolio should allow any reasonably-sized trade."""

    def test_allows_trade_on_empty_portfolio(self, budget: PortfolioRiskBudget) -> None:
        result = budget.check(
            new_strategy="momentum",
            new_symbol="AAPL",
            new_risk_dollars=50.0,
            current_positions=[],
        )
        assert result.is_allowed is True
        assert result.portfolio_var == 0.0
        assert result.portfolio_cvar == 0.0
        assert result.marginal_cvar == 75.0  # 50 * 1.5
        assert result.rejection_reason is None


class TestCVaRBudget:
    """Portfolio CVaR must not exceed the max_portfolio_cvar limit."""

    def test_rejects_when_cvar_exceeds_max(self, budget: PortfolioRiskBudget) -> None:
        # Current positions sum to 300 risk -> CVaR = 450
        positions = [
            {"strategy": "momentum", "symbol": "AAPL", "risk_dollars": 100.0, "side": "long"},
            {"strategy": "mean_revert", "symbol": "MSFT", "risk_dollars": 100.0, "side": "long"},
            {"strategy": "breakout", "symbol": "GOOG", "risk_dollars": 100.0, "side": "long"},
        ]
        # Adding 50 -> total 350 -> CVaR = 525 > 500
        result = budget.check(
            new_strategy="vwap",
            new_symbol="TSLA",
            new_risk_dollars=50.0,
            current_positions=positions,
        )
        assert result.is_allowed is False
        assert "CVaR" in result.rejection_reason
        assert result.portfolio_var == 300.0
        assert result.portfolio_cvar == 450.0

    def test_allows_when_cvar_within_limit(self, budget: PortfolioRiskBudget) -> None:
        positions = [
            {"strategy": "momentum", "symbol": "AAPL", "risk_dollars": 100.0, "side": "long"},
            {"strategy": "mean_revert", "symbol": "MSFT", "risk_dollars": 100.0, "side": "long"},
        ]
        # Adding 30 -> total 230 -> CVaR = 345 < 500
        result = budget.check(
            new_strategy="breakout",
            new_symbol="GOOG",
            new_risk_dollars=30.0,
            current_positions=positions,
        )
        assert result.is_allowed is True
        assert result.rejection_reason is None


class TestStrategyConcentration:
    """No single strategy may exceed max_strategy_risk_pct of total risk."""

    def test_rejects_strategy_concentration_above_40pct(self, budget: PortfolioRiskBudget) -> None:
        positions = [
            {"strategy": "momentum", "symbol": "AAPL", "risk_dollars": 100.0, "side": "long"},
            {"strategy": "mean_revert", "symbol": "MSFT", "risk_dollars": 50.0, "side": "long"},
        ]
        # Adding 100 momentum -> momentum total 200 / 250 = 80% > 40%
        result = budget.check(
            new_strategy="momentum",
            new_symbol="GOOG",
            new_risk_dollars=100.0,
            current_positions=positions,
        )
        assert result.is_allowed is False
        assert "Strategy concentration" in result.rejection_reason


class TestSymbolConcentration:
    """No single symbol may exceed max_symbol_risk_pct of total risk."""

    def test_rejects_symbol_concentration_above_25pct(self, budget: PortfolioRiskBudget) -> None:
        positions = [
            {"strategy": "momentum", "symbol": "AAPL", "risk_dollars": 50.0, "side": "long"},
            {"strategy": "mean_revert", "symbol": "MSFT", "risk_dollars": 50.0, "side": "long"},
        ]
        # Adding 50 AAPL -> AAPL total 100 / 150 = 66.7% > 25%
        result = budget.check(
            new_strategy="breakout",
            new_symbol="AAPL",
            new_risk_dollars=50.0,
            current_positions=positions,
        )
        assert result.is_allowed is False
        assert "Symbol concentration" in result.rejection_reason


class TestAllChecksPassing:
    """When all limits are respected, the trade is allowed."""

    def test_allows_diversified_trade(self, budget: PortfolioRiskBudget) -> None:
        positions = [
            {"strategy": "momentum", "symbol": "AAPL", "risk_dollars": 30.0, "side": "long"},
            {"strategy": "mean_revert", "symbol": "MSFT", "risk_dollars": 30.0, "side": "long"},
            {"strategy": "breakout", "symbol": "GOOG", "risk_dollars": 30.0, "side": "long"},
            {"strategy": "vwap", "symbol": "AMZN", "risk_dollars": 30.0, "side": "long"},
        ]
        # Adding 20 on new strategy/symbol -> total 140 -> CVaR 210 < 500
        # Strategy pct = 20/140 = 14.3% < 40%, Symbol pct = 20/140 = 14.3% < 25%
        result = budget.check(
            new_strategy="gap_fill",
            new_symbol="META",
            new_risk_dollars=20.0,
            current_positions=positions,
        )
        assert result.is_allowed is True
        assert result.rejection_reason is None
        assert result.portfolio_var == 120.0
        assert result.portfolio_cvar == 180.0
        assert result.marginal_cvar == 30.0  # 20 * 1.5


class TestRiskBudgetResultDataclass:
    """RiskBudgetResult is a proper dataclass."""

    def test_default_rejection_reason_is_none(self) -> None:
        r = RiskBudgetResult(is_allowed=True, portfolio_var=0.0, portfolio_cvar=0.0, marginal_cvar=0.0)
        assert r.rejection_reason is None
