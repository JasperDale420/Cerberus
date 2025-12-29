"""
Strategy Library - Signal generation and trading logic.

This package contains all strategy implementations for the Cerberus trading system,
from the abstract base class to concrete VWAP, flow momentum, and breakout strategies.

Components:
    - BaseStrategy: Abstract base class defining strategy interface
    - VWAP Strategies: vwap.py, vwap_reversion.py, vwap_momentum.py
    - Flow Strategies: flow_momentum.py (momentum on unusual options flow)
    - Breakout Strategies: failed_breakout.py (fade failed breakouts)
    - Configuration: config_models.py (Pydantic models for strategy config)

Strategy Lifecycle:
    1. Registration: Strategy registered with StrategyEngine at startup
    2. Bar Events: on_bar() called for each symbol on each new bar
    3. Signal Generation: Strategy emits Signal objects when conditions met
    4. Exit Management: on_bar() also checks for exit conditions (stops/targets)

Development Guide:
    To create a new strategy:
    1. Subclass BaseStrategy from src.strategies.base
    2. Implement on_bar(symbol_state, market_state) -> Optional[Signal]
    3. Add strategy to config.yaml under 'strategies' section
    4. Configure regime routing in 'strategy_routing' section

Key Concepts:
    - Strategies are stateless (state lives in SymbolState)
    - Signals include correlation_id for end-to-end tracking
    - Stop loss/take profit can be in price or ATR multiples
    - Strategies can be regime-specific (bull/bear/neutral/volatile)
    - Feature dependencies declared in FEATURES class attribute

See Also:
    - src.engine: ExecutionEngine routes signals to orders
    - src.data: FeatureCalculator provides technical indicators
    - src.analysis: Regime detection for strategy routing
"""
