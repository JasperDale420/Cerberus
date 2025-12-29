"""
Analysis - Trade tracking, persistence, and analytics.

This package handles all data persistence and post-trade analytics for the
Cerberus trading system, from trade logging to regime detection to performance metrics.

Components:
    - DatabaseDatabase: SQLite persistence with async write buffering
    - Schema: Trade, Fill, Signal, Order, ScannerSnapshot tables
    - Regime: Market regime classification (bull/bear/neutral/volatile)
    - Analytics: Trade metrics, win rate, R-multiple distribution

Database Schema:
    - Trade: Completed round-trip trades with PnL and metrics
    - Fill: Individual fill events from broker
    - Signal: Strategy signals (accepted and rejected)
    - Order: Order lifecycle tracking
    - ScannerSnapshot: Historical watchlist snapshots

Analytics Capabilities:
    - Trade metrics: Win rate, average R, max drawdown
    - Regime analysis: Performance by market regime
    - Strategy evaluation: Per-strategy P&L and statistics
    - Temporal analysis: Time-of-day, day-of-week patterns

Key Concepts:
    - Best-effort database writes (trading continues on DB failure)
    - Write buffering for performance (PRD 11.4)
    - Regime detected via VIX and market breadth
    - All timestamps stored in UTC
    - Correlation IDs link signals → orders → fills → trades

Database Fault Tolerance (PRD 11.4):
    - db_fail_mode: warn|error|raise (how to handle write failures)
    - db_write_buffer: In-memory queue for transient outages
    - db_trading_mode: continue|pause (trading behavior on DB failure)

See Also:
    - src.engine: ExecutionEngine persists trades and fills
    - src.strategies: Signal generation feeds analytics
"""
