# Portfolio Layer

The portfolio layer sits between strategy signal generation and order execution, providing risk-aware allocation and performance tracking.

## Signal Flow (Intraday)

```
Strategy.evaluate_bar()
    → SignalAggregator.combine()       # IC-weighted signal combination
    → RiskManager.check_signal()       # existing risk gates
    → PortfolioRiskBudget.check()      # VaR/CVaR + concentration limits
    → OrderExecutor.submit()
```

## EOD Flow

```
PortfolioAllocator.recompute()         # risk-parity allocation with drawdown throttle
PortfolioPerformance.update()          # strategy attribution, rolling Sharpe/Sortino
```

## Integration Note

Wiring these components into `ExecutionEngine` (src/engine/execution.py) is a separate PR to avoid breaking the live pipeline. The modules are designed as drop-in additions — they read from existing position/trade data and write to new DB tables (`strategy_ic_daily`, `portfolio_risk_snapshots`).
