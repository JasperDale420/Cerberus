"""
Data Pipeline - Market data fetching and feature calculation.

This package handles all data acquisition and technical indicator calculation
for the Cerberus trading system, from raw OHLCV bars to computed features.

Components:
    - AlpacaClient: Market data API and account information (singleton)
    - Fetcher: Historical data retrieval with caching
    - FeatureCalculator: Technical indicator calculation (VWAP, ATR, RSI, etc.)
    - Repository: Historical data persistence and retrieval

Data Pipeline:
    Raw Bars (Alpaca) → FeatureCalculator → Technical Indicators → Strategy

Key Concepts:
    - AlpacaClient is a singleton (one instance per process)
    - Historical data cached in SQLite for performance
    - Real-time bars received via WebSocket, features calculated on-the-fly
    - Feature calculator maintains rolling windows for indicators
    - Indicator cache cleared daily to prevent memory growth

Feature Types:
    - Price-based: VWAP, pivots, high/low of day
    - Volatility: ATR (Average True Range), Bollinger Bands
    - Momentum: RSI (Relative Strength Index), rate of change
    - Volume: Volume profile, VWAP deviation

Performance:
    - Features cached per symbol to avoid recomputation
    - Historical data fetched in bulk for backtesting
    - Real-time: incremental updates only for active watchlist

See Also:
    - src.strategies: Consumers of calculated features
    - src.engine: ExecutionEngine coordinates data flow
    - src.analysis: Database schema for historical data persistence
"""
