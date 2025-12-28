# Cerberus Trading System

**Current Version**: 1.0.0 (Dec 2025)

Cerberus is an automated, modular algorithmic trading system designed for both live and paper trading on US equities. It integrates real-time market data from Alpaca and options flow from Unusual Whales to drive a suite of technical and flow-based strategies. The system features a robust execution engine, a comprehensive feature pipeline, and an agentic analytics layer for daily performance review.

## Current Capabilities

- **Live & Paper Trading**: Seamless switching between paper simulation and live execution via `--mode`.
- **Modular Strategy Engine**: Plug-and-play strategy support including:
  - VWAP Reversion & Trend Rider
  - Opening Range Breakout (ORB)
  - Trend Pullback
  - Failed Breakout
  - Gap Fill
  - Flow Momentum & Index Mean Reversion
- **Advanced Data Pipeline**:
  - Real-time bar aggregation and technical indicator calculation (Pandas-TA).
  - Unusual Whales options flow integration.
  - Multi-stage scanner with data quality gates.
- **Resilient Execution**:
  - Automated market open/close handling.
  - End-of-Day (EOD) position flattening (`flat-on-close`).
  - "Noop" executor mode for safe logic verification.
- **Agentic Analytics**: Automated EOD performance analysis and database aggregation.
- **Internal Scheduler**: Native `APScheduler` integration for persistent process management (replaces external cron).

## Non-Capabilities / Explicit Non-Goals

- **High-Frequency Trading (HFT)**: Not designed for microsecond-latency arbitrage.
- **Crypto/Forex**: strictly US Equities focused.
- **Multi-Broker**: Currently tightly coupled to Alpaca for execution.
- **Overnight Holds**: Strictly intraday; all positions are closed at 16:00 ET.

## Architecture Overview

Cerberus follows a vertical-slice architecture optimized for reliability and testability:

1.  **Scanner (`src/scanner`)**: Filters the universe of 8000+ tickers down to actionable candidates using technicals and flow.
2.  **Feature Pipeline (`src/data`)**: Ingests raw data (Alpaca/UW), computes features, and caches results.
3.  **Strategy Engine (`src/strategies`)**: Evaluates candidates against active strategies to generate Signals.
4.  **Execution Engine (`src/engine`)**: Converts Signals into Orders, manages Risk (sizing/limits), and handles Fills.
5.  **Analytics (`src/analysis`)**: Persists trade state to SQLite (`cerberus.db`) and generates reports.

## Repository Structure

```
├── .github/                # CI/CD workflows
├── artifacts/              # Generated reports and logs
├── config/                 # Application configuration (YAML)
├── scripts/                # Utility scripts (ingestion, manual loops)
├── src/
│   ├── analysis/           # Database, Schema, Analytics Engine
│   ├── data/               # Data Fetchers, Pipeline, Alpaca/UW Clients
│   ├── engine/             # Execution Engine, Order Management
│   ├── scanner/            # Universe Selection, Profiles, Validation
│   ├── strategies/         # Strategy Implementations
│   ├── main.py             # Application Entry Point
│   └── scheduler.py        # Internal Scheduler Service
├── tests/                  # Pytest Suite (Unit, Integration, E2E)
├── Dockerfile              # Container definition
├── Makefile                # Dev automation commands
├── pyproject.toml          # Tool configuration (Ruff, Mypy, Pytest)
└── requirements.txt        # Python dependencies
```

## Setup & Installation

### Prerequisites
- Python 3.11+
- SQLite3
- Alpaca API Credentials (Paper or Live)
- Unusual Whales API Token (Optional, for flow strategies)

### Local Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/EmpireTrading/Cerberus.git
    cd Cerberus
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Configuration**:
    Create a `.env` file or export variables:
    ```bash
    export APCA_API_KEY_ID="<your_key>"
    export APCA_API_SECRET_KEY="<your_secret>"
    export UNUSUAL_WHALES_API_TOKEN="<your_token>"
    ```

### Docker Setup
1.  **Build image**:
    ```bash
    docker build -t cerberus .
    ```
2.  **Run container**:
    ```bash
    docker run --env-file .env cerberus
    ```

## How to Run

### Development / Paper Trading
Run a single pass of the system in paper mode:
```bash
python -m src.main --mode paper --run-once
```

Run the continuous trading loop (paper):
```bash
python -m src.main --mode paper
```

### Production / Live Trading
**WARNING**: Real money risk. Ensure `APCA_API_BASE_URL` points to live API.
```bash
python -m src.main --mode live --config config/prod.yaml
```

### Persistent Scheduler
Run as a background service that auto-starts trading at market open:
```bash
python -m src.main --scheduler
```

### Utility Commands
- **Ingest SEC Tickers**: `python scripts/ingest_sec_tickers.py`
- **Run EOD Analytics**: `python -m src.main --eod`

### Backtesting
Run a historical replay using the full `ExecutionEngine` (portfolio-style, across all symbols):
```bash
python scripts/run_backtest.py --config config/config.yaml --start-date 2024-01-03 --end-date 2024-01-10
```

For deterministic offline runs (no Alpaca historical fetch), provide JSONL bars:
```bash
python scripts/run_backtest.py --config config/config.yaml --start-date 2024-01-03 --end-date 2024-01-10 --offline-bars-dir /path/to/bars
```
Bars directory format: `{SYMBOL}_{timeframe}.jsonl` (example: `AAPL_1Min.jsonl`) with one JSON object per line:
`{"t":"2025-01-01T14:30:00+00:00","o":1,"h":1,"l":1,"c":1,"v":100}`.

Accuracy notes:
- Backtests enforce **flat-at-close** by flattening at the end of each session (robust even when your dataset has no `16:00` bar).
- Optional realism knobs (in `risk:`): `slippage_bps`, `spread_bps`, `commission_per_share`, `min_commission`.
- Optional stale order cancel (top-level): `max_open_order_age_sec`.
- If `scanner.enabled: true` and `scanner.interval_minutes > 0`, backtests replay scanner gating at that cadence using technical features computed from the loaded bars (flow features are neutral unless you implement an offline flow source).
- Output keys:
  - `metrics`: engine-native summary derived from closed trades
  - `engine_trades`: per-trade records (net/gross, MAE/MFE, holding time, etc.)
  - `metrics_fills`: legacy fill-based analyzer output (useful for debugging but less accurate)

## Configuration

Configuration is managed via `config/config.yaml` and environment variables. Key sections:

-   **`trading`**: Risk limits, leverage, max positions.
-   **`scanner`**: Universe filters, min volume/price.
-   **`strategies`**: Enable/disable specific strategies and tune parameters.
-   **`logging`**: Log levels and formatting.

## Error Handling & Logging

-   **Structured Logging**: JSON-formatted logs written to `logs/` and stdout.
-   **Error Policy**: Fails fast on startup/config errors. During trading, catches strategy exceptions to prevent system crash, logging them as `ERROR` while maintaining the main loop.
-   **Observability**: Key metrics (fetch failures, scan times) are logged for monitoring.

## Testing Status

The project maintains high test coverage via `pytest`.

-   **Unit Tests**: `tests/unit/` - Core logic verification.
-   **Integration Tests**: `tests/integration/` - Database and Component interaction.
-   **E2E Tests**: `tests/e2e/` - Full flow validation with mocked Broker.

**Run all tests**:
```bash
pytest
```
**Run with coverage**:
```bash
pytest --cov=src
```

## Safety & Risk Notes

-   **Noop Executor**: Use `--order-executor noop` to verify logic without sending orders to Alpaca.
-   **Flatten on Close**: The main loop enforces a hard exit at 16:00 ET, attempting to close all positions.
-   **Database Buffer**: Writes are buffered to SQLite to handle high-throughput periods, but this implies a potential (small) data loss risk on hard crash.

## Known Gaps & TODOs

-   **Data Quality**: Scanner validation is implemented but could be expanded for complex flow anomalies.
-   **Backfill**: Historical data backfill tooling is basic (`scripts/ingest_sec_tickers.py` restored, but full OHLCV backfill is manual).
-   **LLM Integration**: `LLMClient` stubbed in code; full agentic reasoning loop is in early stages ("Agent Stage 1").

## Contribution Notes

1.  **Vertical Slices**: Implement complete features (API -> DB), not horizontal layers.
2.  **Testing**: New features must include unit tests. Run `make lint` and `pytest` before PR.
3.  **Docs**: Update `CHANGELOG.md` upon merging.
