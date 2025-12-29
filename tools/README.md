# Utility Tools

This directory contains development and verification utilities for the Cerberus trading system.

## Available Tools

### `verify_architecture.py`
**Purpose**: Validates the repository structure and key architectural patterns.

**Usage**:
```bash
python tools/verify_architecture.py
```

**What it checks**:
- Required directories exist (`src/`, `tests/`, `config/`)
- Core modules are present (data, engine, strategies, scanner, analysis)
- Key configuration files are in place

### `verify_deepseek.py`
**Purpose**: Tests connectivity and basic functionality of the DeepSeek LLM API endpoint.

**Usage**:
```bash
python tools/verify_deepseek.py
```

**Requirements**:
- `DEEPSEEK_API_KEY` or `CENTRAL_LLM_API_URL` environment variable
- Running LLM service (if using local deployment)

### `paper_live_harness.py`
**Purpose**: Interactive test harness for paper trading validation and live trading dry-runs.

**Usage**:
```bash
python tools/paper_live_harness.py [--config config/config.yaml]
```

**Features**:
- Simulates full trading loop in paper mode
- Validates strategy execution without real capital risk
- Useful for pre-deployment testing

## When to Use These Tools

- **Before committing**: Run `verify_architecture.py` to ensure structural integrity
- **Before deploying**: Run `paper_live_harness.py` to validate trading logic
- **LLM integration**: Use `verify_deepseek.py` to test agentic analytics connectivity

## Maintenance

These utilities are for development purposes only and are not part of the core trading system. They may have looser quality standards than production code but should remain functional.
