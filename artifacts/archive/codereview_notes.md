# Code Review Notes

## 1. Inventory
- **Runtime:** Python 3.10+, Asyncio.
- **Entry Points:** `src/main.py` (Live/Paper), `src/scheduler.py` (Scheduler).
- **Core Components:**
    - `ExecutionEngine`: Central orchestrator.
    - `Scanner`: Universe selection & filtering.
    - `FeaturePipeline`: Data fetching/computation.
    - `StrategyEngine`: Signal generation.
    - `RiskManager`: Risk checks.
    - `OrderExecutor`: Broker interaction.
    - `DatabaseDatabase`: Persistence (SQLite/Postgres abstraction?).

## 2. Trace: Live Trading
- **Flow:**
    1. `main.py` initializes components.
    2. `ExecutionEngine` starts.
    3. `alpaca_client` streams bars -> `engine.on_bar()`.
    4. `on_bar` -> `RegimeDetector` (if index) -> `StrategyEngine.on_bar` -> `Signal`.
    5. `Signal` -> `_process_signal` -> `RiskManager.apply` -> `intents`.
    6. `intents` -> **[MISSING LINK]** Order submission? (Need to verify `_process_signal` end).
    7. `Scanner` runs periodically to update... what? Watchlist? Subscriptions?
        - `Scanner.scan()` returns `ScanResult` with `watchlist`.
        - Does `ExecutionEngine.run_scan` update Alpaca subscriptions? **[VERIFY]**

## 3. Findings (Draft)

### A. Correctness
- `Scanner.scan_async` calls `self.scan` which calls `self.feature_pipeline.compute_technicals_only`.

### B. Architecture
- Vertical Slices:
    - `src/scanner`, `src/strategies`, `src/engine` seem reasonable.
    - `src/data` is shared infra.

### C. Observability
- `StructuredLogger` used extensively.
- `correlation_id` seems to be passed around.
- Health metrics in `ExecutionEngine`.

### D. Security
- Secrets loading via `ConfigLoader`?
- `UnusualWhalesClient` likely has API key. `AlpacaClient` too.

### E. Reliability
- `main.py` has a global try/except block.
- `engine.flatten_all` has robust error handling options (`mismatch_mode`).

### F. Testing
- `tests` folder has 56 files. Coverage looks active (`.coverage` file exists).

### G. Documentation
- `PRD.md` is huge (31KB). `README.md` is active.

## Todo Questions
- How do we handle dynamic subscriptions based on Scanner results?
- Where is the order actually sent in `_process_signal`?
- How does Backtesting work?
