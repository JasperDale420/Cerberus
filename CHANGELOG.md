# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Strategies**: Implemented full suite of 8 remediation strategies:
    - VWAP Mean Reversion
    - Opening Range Breakout (ORB)
    - Trend Pullback
    - Failed Breakout Fade
    - VWAP Trend Rider
    - Index Mean Reversion
    - Flow-Confirmed Momentum
    - Gap-Fill Scalper
- **Scanner**: Implemented `ScannerProfile` interface and specific profiles for all 8 strategies. Filters based on technicals (ADX, RSI, BB) and Option Flow (Unusual Whales Z-Score).
- **Pipeline**: Added comprehensive feature generation:
    - `prior_day_high`, `prior_day_low`
    - `bb_upper`, `bb_lower`, `price_zscore`
    - `flow_zscore`, `call_put_ratio` (Unusual Whales)
    - `premarket_volume` calculation
- **Architecture**:
    - `Agent` meta-loop for daily analysis and config updates.
    - `Analytics` layer for trade statistics and efficiency auditing.
    - `Scheduler` integration for automated functionality.
- **Testing**: Added unit tests for all strategies (`tests/test_strategy_*.py`).

### Changed
- **Scanner Core**: Fixed duplicate watchlist entry bug and added sorting by score.
- **Pipeline**: Removed hardcoded `premarket_volume`; now calculates from intraday data.
- **Config**: Extended `config.yaml` to support all new strategies and parameters.

### Fixed
- Addressed various linting issues (Cognitive Complexity updates pending major refactor).
