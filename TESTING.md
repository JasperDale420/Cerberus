# Testing Guide

This repository uses `pytest` for all testing. We aim for high coverage and deterministic reliability.

## Quickstart

Run the full suite as CI does:
```bash
make test-ci
```

Or run interactively during development:
```bash
# Run unit tests only
make test-unit

# Run end-to-end smoke tests
make test-e2e
```

## Directory Structure

```
tests/
├── unit/           # Isolated component tests (ExecutionEngine, RiskManager, etc.)
├── integration/    # Component interaction tests (DB persistence, file I/O)
├── contract/       # API contract verification with mocked transports
├── e2e/            # Full system smoke tests (Scheduler -> Engine -> DB)
├── conftest.py     # Shared fixtures (DB init, Mock Alpaca, Config helpers)
└── benchmark_...   # Performance benchmarks (excluded from standard CI)
```

## Test Pyramid & Strategy

1.  **Unit Tests (70% of volume)**
    *   **Goal**: Verify logic of classes/functions in isolation.
    *   **Constraint**: NO I/O (no DB, no Network). Use `MagicMock`.
    *   **Location**: `tests/unit/`

2.  **Integration Tests (20% of volume)**
    *   **Goal**: Verify component wiring (e.g., proper SQL queries, config loading).
    *   **Constraint**: Use generic/tempfile I/O. Use SQLite in memory/tempfile.
    *   **Location**: `tests/integration/`

3.  **End-to-End (E2E) (10% of volume)**
    *   **Goal**: Validate critical "Vertical Slices" (Signal -> Order -> Fill -> Analytics).
    *   **Constraint**: No live broker connection; offline simulation only.
    *   **Location**: `tests/e2e/`

## Coverage Policy

*   **Threshold**: CI enforces **70%** code coverage.
*   **Exclusions**: `src/main.py` (entrypoint), `src/ui/`.
*   **Report**: `make test-ci` generates `coverage.xml` and an HTML report.

## CI Integration

CI runs on every push to `main` and all PRs.
*   **Workflows**: `.github/workflows/ci.yml`
*   **Artifacts**: JUnit XML results and Coverage XML are uploaded.

## Development Rules

1.  **Determinism**: Tests must not rely on `datetime.now()` directly (inject clocks). Tests must not hit real APIs.
2.  **Performance**: The entire suite should run in under 30s locally.
3.  **Isolation**: Use `tmp_path` fixture for files; do not write to the source tree.

