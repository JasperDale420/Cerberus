"""
Scanner - Universe selection and symbol filtering.

This package implements dynamic watchlist construction based on real-time
market data and configurable filters for tradability.

Components:
    - Scanner: Main scanner orchestrating universe selection, validation, and ranking
    - UniverseBuilder: 3-tier symbol selection (explicit + static + dynamic volume)
    - DataValidator: Baseline price/volume/ATR filters
    - RankingEngine: Cross-sectional alpha scoring
    - PairScanner: Cointegration-based pair discovery
    - StreamingScanner: Real-time OFI/TFI microstructure updates

Scanner Workflow:
    1. Build universe (explicit symbols + static files + dynamic volume)
    2. Fetch technicals and apply data validation filters
    3. Fetch flow data and rank symbols cross-sectionally
    4. Assign strategies via strategy_routing config by regime
    5. Build watchlist sorted by AlphaScore

Configuration (config.yaml):
    - scanner: interval, max_watchlist_size, validation thresholds
    - universe: explicit symbols, static files, dynamic volume sources
    - strategy_routing: per-regime strategy assignments

See Also:
    - src.engine: ExecutionEngine consumes ScanResult
    - src.data: FeatureCalculator provides volatility metrics
"""
