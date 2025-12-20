# Cerberus: Multi-Strategy Intraday Scalping System

## 1. Project Overview
Cerberus is a deterministic, vertical-slice-architected intraday trading system tailored for US equities. It integrates **Alpaca** for real-time data and execution with **Unusual Whales** for options flow-based signal enhancement. The system is designed to run multiple plug-and-play strategies simultaneously across a dynamic watchlist, leveraging a market regime classifier (Bull, Bear, Chop) to adapt strategy behavior in real-time. It prioritizes "fail-fast" engineering, comprehensive logging, and agentic continuous improvement.

## 2. Current Capabilities
- **Multi-Strategy Execution Engine**: Runs deterministic strategies like `VWAP Reversion`, `Trend Pullback`, `ORB`, and `Gap Fill`.
- **Market Regime Detection**: Real-time classification of market state (BULL, BEAR, CHOP) using rolling SPY returns and volatility.
- **Data Pipeline**:
  - Real-time bar aggregation via Alpaca WebSocket.
  - Options flow analysis via Unusual Whales API (flow scores, call/put ratios).
  - Feature engineering (ATR, RSI, Z-scores, Relative Volume).
- **Scanner**: Periodically scans the universe to update the active watchlist based on strategy-specific criteria.
- **Risk Management**:
  - Account-level checks (daily loss limit, max open risk).
  - Strategy-level limits (max R per trade).
  - Symbol-level exposure caps.
- **Analytics & Persistence**:
  - SQLite database (`cerberus.db`) storing trades, signals, orders, fills, and regime history.
  - End-of-Day (EOD) aggregations for performance tracking.
- **Agentic Loop (Stage 1)**: Automated analysis of daily performance to adjust risk limits strategies.

## 3. Non-Capabilities / Explicit Non-Goals
- **Overnight Holds**: Strictly intraday. All positions are flattened at market close (16:00 ET).
- **Options Trading**: The system does NOT trade options contracts; it only consumes options flow data as a signal for equities.
- **Generative AI Strategy Creation**: Strategies are currently code-based and deterministic, not hallucinated by LLMs in real-time.
- **Docker/Containerization**: Container support is currently **missing/inprogress** (see Known Gaps).

## 4. Architecture Overview
The system follows a **vertical slice** architecture where features cut across these layers:

- **Config Layer**: YAML-based configuration for strategies, risk, and universe.
- **Data Layer (`src/data`)**: Adapters for Alpaca and Unusual Whales.
- **Scanner (`src/scanner`)**: Filters universe -> calculates features -> ranks symbols.
- **Engine (`src/engine`)**: Event loop handling bars -> Strategy Engine -> Signals -> Risk Manager -> Order Executor.
- **Strategy Layer (`src/strategies`)**: Isolated, stateless logic classes implementing `BaseStrategy`.
- **Analysis (`src/analysis`)**: DB access, analytics engine, and agentic feedback loop.

Data flows from **Market Data** -> **Engine** -> **Strategies** -> **Signals** -> **Risk** -> **Execution**.

## 5. Repository Structure
```
├── config/                 # YAML Configuration files
├── patches/                # Agentic code patches
├── scripts/                # Helper scripts for paper/continuous runs
├── src/
│   ├── agent/              # Agentic loop logic
│   ├── analysis/           # Database and Analytics engine
│   ├── backtest/           # Backtesting harness (stub/wip)
│   ├── core/               # Shared domain models (Bar, Signal, etc.)
│   ├── data/               # Alpaca/Unusual Whales clients & pipeline
│   ├── engine/             # Execution loop, Risk Manager, Order Executor
│   ├── scanner/            # Universe selection and feature generation
│   ├── strategies/         # Strategy implementations (VWAP, ORB, etc.)
│   └── main.py             # Application entry point
├── tests/                  # Pytest suite
│   ├── unit/               # Fast, isolated tests
│   ├── integration/        # Data/DB integration tests
│   └── e2e/                # Full flow tests
├── Makefile                # Command shortcuts
├── PRD.md                  # Detailed Product Requirements Document
├── requirements.txt        # Python dependencies
├── DEVELOPER_NOTES.md      # IDE and workflow tips
└── CONTRIBUTING.md         # Contribution guidelines
```

## 6. Setup & Installation
1.  **Prerequisites**: Python 3.11+, Make.
2.  **Environment Setup**:
    ```bash
    # Create virtual environment (optional but recommended)
    python -m venv .venv
    source .venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Install dev tools and pre-commit hooks
    pip install pre-commit
    pre-commit install
    ```
3.  **Credentials**: 
    Create a `.env` file in the root directory (see `.env.example` if available, or PRD/Config refs) containing:
    ```env
    ALPACA_API_KEY=...
    ALPACA_SECRET_KEY=...
    UNUSUAL_WHALES_API_KEY=...
    # Optional overrides
    LOG_LEVEL=INFO
    ALPACA_PAPER=true
    ```

## 7. How to Run

### Local Paper Trading
Run the main entry point:
```bash
python -m src.main --mode paper --config config/config.yaml
```

### Verification Mode (Run Once)
Run a single scan cycle and exit to verify connectivity and logic:
```bash
python -m src.main --mode paper --run-once
```

### Tests
Use the Makefile for standard testing workflows:
```bash
make test          # Run all tests with coverage
make test-unit     # Run fast unit tests
make test-e2e      # Run end-to-end smoke tests
```

## 8. Configuration
Configuration is driven by YAML files in `config/`:
- **config.yaml**: Main system config (API keys, global settings, logging).
- **strategies.yaml**: Enable/disable strategies and tune parameters (e.g., `sigma_band`, `risk_reward`).
- **risk.yaml**: Define risk limits (e.g., `max_daily_loss`, `max_order_size`).
- **universe.yaml**: Source lists for the scanner (e.g., SP500, NASDAQ100).

Overrides can be provided via `.env` variables or `strategies.auto.yaml` (generated by the Agent).

## 9. Error Handling & Logging
- **Structured Logging**: All logs are JSON-structured (or structured key-value) via `StructuredLogger`.
- **Log Levels**:
  - `INFO`: Normal operation (state changes, orders).
  - `WARNING`: Recoverable issues (API timeouts, skipped symbols).
  - `ERROR`: Logic failures or critical I/O errors.
- **Fail Fast**: The system is designed to crash on startup configuration errors or critical infrastructure failures, rather than running in an invalid state.
- **Output**: Logs are printed to stdout and can be configured to write to `logs/` directory.

## 10. Testing Status
- **Framework**: `pytest` with `pytest-cov`.
- **Coverage**: High coverage enforced (>70% global, aimed at 100% for core logic).
- **Categories**:
  - **Unit**: Strategy logic, domain models, risk calculations.
  - **Integration**: Database persistence, Client adapters (mocked I/O).
  - **E2E**: Full system loops using `paper_live_harness.py`.
- **CI**: GitHub Actions runs `make test-ci` on PRs.

## 11. Safety & Risk Notes
- **Intraday Only**: Ensure the process triggers the "flatten all" logic at 16:00 ET.
- **Risk Limits**: Always verify `risk.yaml` limits before live trading.
- **Paper Trading**: Default mode is `paper`. Explicitly set `--mode live` only when ready to deploy capital.
- **Stop Losses**: Strategies generate stop prices, but ensure the `OrderExecutor` is correctly handling `stop_loss` parameters if supported by the broker, or that the engine is monitoring them reliably.

## 12. Known Gaps & TODOs
- **Docker**: No `Dockerfile` or `docker-compose.yml` present in the root. Containerization is currently manual or missing.
- **Backtesting**: `src/backtest` exists but `runner.py` and full historical simulation capabilities may be experimental or incomplete compared to the live engine.
- **Options Execution**: Not implemented (Scope: Equities only).

## 13. Contribution Notes
- **Vertical Slices**: Submit PRs that implement full features (Config -> Data -> Logic -> Test), not horizontal layers.
- **Linting**: Strict `ruff`, `black`, and `mypy` enforcement. Run `make ci` before pushing.
- **Commit Messages**: Use clear, descriptive comments.
- **Documentation**: Update this README or `PRD.md` if architectural changes are made.
