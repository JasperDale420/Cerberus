"""
Scanner - Universe selection and symbol filtering.

This package implements dynamic watchlist construction based on real-time
unusual options flow data and configurable filters for tradability.

Components:
    - Scanner: Main scanner orchestrating filters and ranking
    - Filters: Volume, price range, volatility, and flow-based filters
    - Ranking: Prioritization logic for most attractive symbols

Scanner Workflow:
    1. Fetch unusual options flow from Unusual Whales API
    2. Apply filters (volume > threshold, price in range, etc.)
    3. Rank symbols by flow strength and other criteria
    4. Return top N symbols as ScanResult for watchlist

Filters:
    - Minimum volume: Ensure liquidity for execution
    - Price range: Avoid penny stocks and ultra-expensive stocks
    - Volatility: Filter out low-volatility symbols
    - Flow strength: Prioritize strong unusual options activity
    - Existing positions: Optionally exclude already-held symbols

Configuration (config.yaml):
    - scanner_interval: How often to refresh (bars)
    - max_watchlist_size: Maximum symbols in watchlist
    - filters: Volume, price, ATR thresholds
    - ranking: Weights for flow vs momentum vs volatility

Key Concepts:
    - Scanner runs periodically (not every bar) for performance
    - Results cached to avoid redundant API calls
    - Watchlist can change intraday based on flow updates
    - PRD 6.3: Scanner integration with execution engine

See Also:
    - src.engine: ExecutionEngine consumes ScanResult
    - src.data: FeatureCalculator provides volatility metrics
"""
