"""
Execution Engine - Order execution and position management.

This package contains the core execution logic for the Cerberus trading system,
orchestrating the flow from strategy signals to order execution to position tracking.

Components:
    - ExecutionEngine: Main execution loop, bar processing, signal routing
    - OrderExecutor: Order submission, cancellation, fill handling
    - PositionManager: Position tracking, PnL calculation, exit logic
    - RiskManager: Pre-trade risk checks, position/loss limits, sizing

Data Flow:
    Strategy Signal → RiskManager → OrderExecutor → Broker → Fill → PositionManager

Key Concepts:
    - All positions are intraday (closed at 16:00 ET or on risk breach)
    - Orders tracked via correlation_id for end-to-end observability
    - PnL tracked in both $ and R-multiples for risk management
    - Bracket orders (stop/take profit) can be delegated to broker (PRD 6.7)
    - Best-effort semantics for database writes and observability

Architecture:
    ExecutionEngine coordinates all components and manages the main trading loop.
    It receives bars from the data feed, routes them to strategies, processes
    signals through risk checks, submits orders, handles fills, and manages
    positions including exits.

See Also:
    - src.strategies: Signal generation and strategy implementations
    - src.analysis: Trade persistence, analytics, and regime detection
    - src.data: Market data, features, and technical indicators
"""
