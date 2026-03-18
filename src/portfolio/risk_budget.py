"""Portfolio-level risk management with VaR/CVaR and marginal risk contribution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskBudgetResult:
    """Result of a portfolio risk budget check."""

    is_allowed: bool
    portfolio_var: float
    portfolio_cvar: float
    marginal_cvar: float
    rejection_reason: str | None = None


class PortfolioRiskBudget:
    """Portfolio risk budget enforcing VaR/CVaR limits and concentration checks.

    Uses simplified risk proxies:
    - VaR = sum of position risk_dollars
    - CVaR = 1.5x VaR (in practice would use GARCH or historical simulation)
    - Marginal CVaR = new_risk_dollars as fraction of portfolio CVaR
    """

    CVAR_MULTIPLIER = 1.5

    def __init__(
        self,
        max_portfolio_cvar: float = 500.0,
        max_marginal_cvar_pct: float = 0.25,
        max_strategy_risk_pct: float = 0.40,
        max_symbol_risk_pct: float = 0.25,
    ) -> None:
        self.max_portfolio_cvar = max_portfolio_cvar
        self.max_marginal_cvar_pct = max_marginal_cvar_pct
        self.max_strategy_risk_pct = max_strategy_risk_pct
        self.max_symbol_risk_pct = max_symbol_risk_pct

    def check(
        self,
        new_strategy: str,
        new_symbol: str,
        new_risk_dollars: float,
        current_positions: list[dict],
        account_equity: float = 100_000.0,
    ) -> RiskBudgetResult:
        """Check whether a new position is allowed under portfolio risk limits.

        Args:
            new_strategy: Strategy requesting the trade.
            new_symbol: Ticker symbol for the new position.
            new_risk_dollars: Dollar risk of the proposed position.
            current_positions: List of dicts with keys: strategy, symbol, risk_dollars, side.
            account_equity: Total account equity (used for reference, not directly in calcs).

        Returns:
            RiskBudgetResult with approval status and risk metrics.
        """
        # Current portfolio risk
        current_var = sum(p["risk_dollars"] for p in current_positions)
        current_cvar = current_var * self.CVAR_MULTIPLIER

        # Prospective portfolio risk (including new position)
        prospective_var = current_var + new_risk_dollars
        prospective_cvar = prospective_var * self.CVAR_MULTIPLIER

        # Marginal CVaR contribution
        marginal_cvar = new_risk_dollars * self.CVAR_MULTIPLIER

        # --- Concentration checks (use prospective totals) ---
        total_risk = current_var + new_risk_dollars

        if total_risk > 0 and current_positions:
            # Strategy concentration
            strategy_risk = sum(p["risk_dollars"] for p in current_positions if p["strategy"] == new_strategy)
            strategy_risk += new_risk_dollars
            strategy_pct = strategy_risk / total_risk

            if strategy_pct > self.max_strategy_risk_pct:
                return RiskBudgetResult(
                    is_allowed=False,
                    portfolio_var=current_var,
                    portfolio_cvar=current_cvar,
                    marginal_cvar=marginal_cvar,
                    rejection_reason=(
                        f"Strategy concentration {strategy_pct:.1%} exceeds limit {self.max_strategy_risk_pct:.0%}"
                    ),
                )

            # Symbol concentration
            symbol_risk = sum(p["risk_dollars"] for p in current_positions if p["symbol"] == new_symbol)
            symbol_risk += new_risk_dollars
            symbol_pct = symbol_risk / total_risk

            if symbol_pct > self.max_symbol_risk_pct:
                return RiskBudgetResult(
                    is_allowed=False,
                    portfolio_var=current_var,
                    portfolio_cvar=current_cvar,
                    marginal_cvar=marginal_cvar,
                    rejection_reason=(
                        f"Symbol concentration {symbol_pct:.1%} exceeds limit {self.max_symbol_risk_pct:.0%}"
                    ),
                )

        # --- Budget check ---
        if prospective_cvar > self.max_portfolio_cvar:
            return RiskBudgetResult(
                is_allowed=False,
                portfolio_var=current_var,
                portfolio_cvar=current_cvar,
                marginal_cvar=marginal_cvar,
                rejection_reason=(
                    f"Portfolio CVaR ${prospective_cvar:.2f} would exceed limit ${self.max_portfolio_cvar:.2f}"
                ),
            )

        return RiskBudgetResult(
            is_allowed=True,
            portfolio_var=current_var,
            portfolio_cvar=current_cvar,
            marginal_cvar=marginal_cvar,
        )
