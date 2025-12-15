# Cerberus 🐺

**Multi-Strategy Intraday Scalping System**

Cerberus is a deterministic, vertical-slice trading system designed for US Equities. It integrates **Alpaca** for market data/execution and **Unusual Whales** for options flow signals.

## System Overview

- **Scanner**: Filters universe based on technicals and options flow.
- **Engine**: Event-driven architecture processing bars and signals.
- **Risk Manager**: centralized risk enforcement (Daily Loss, Max Position, Risk-per-trade).
- **Execution**: Algo-ready order routing with retry logic.
- **Strategies**: Plug-and-play strategy architecture.
- **Agentic Loop**: Designed for continuous improvement via feedback loops.

## Supported Strategies
1. **VWAP Reversion**: Mean reversion to volume-weighted average price.
2. **ORB (Opening Range Breakout)**: 15-min opening range volatility breakout.
3. **Trend Pullback**: Retracement entry in established trends (ADX+RSI).
4. **Failed Breakout Fade**: Fading false breakouts of Prior Day High/Low.
5. **VWAP Trend Rider**: Trend following on VWAP reclaims with volume.
6. **Index Mean Reversion**: Fading extreme Bollinger Band moves on Indices.
7. **Flow-Confirmed Momentum**: Momentum entries backed by institutional option flow.
8. **Gap-Fill Scalper**: Fading morning gaps that fail to extend.

## Development Workflow

### Prerequisites
- Python 3.11+
- Pip/Poetry
- Alpaca Paper Account (API Key & Secret)

### Setup
1. **Infrastructure**:
   ```bash
   pip install -r requirements.txt
   pip install pre-commit
   pre-commit install
   ```

2. **Environment**:
   Create a `.env` file:
   ```bash
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
   ALPACA_PAPER=true
   ```

### Quality Checks
- **Run all checks**: `pre-commit run --all-files`
- **Run Tests**: `pytest`

## Paper-Live Verification

The repository includes a production-grade test harness `paper_live_harness.py` to verify system behavior in a "Paper-Live" environment (Real Data + Paper Execution).

### Verification Scenarios

**1. Happy Path** (`--scenario happy`)
Runs the bot, injects a valid signal, and verifies order submission.
```bash
python paper_live_harness.py --scenario happy --duration 5 --inject-signal
```

**2. Failure Injection** (`--scenario failure`)
Simulates broker errors (500, 429, Timeout) to verify resilience.
```bash
python paper_live_harness.py --scenario failure --duration 5 --inject-signal
```

**3. Risk Breach** (`--scenario risk`)
Attempts to trade with artificially lowered limits to verify Risk Manager rejection.
```bash
python paper_live_harness.py --scenario risk --duration 5 --inject-signal
```

### Artifacts
Test runs generate evidence in the `artifacts/` directory, including:
- Structured Logs (`.jsonl`)
- Broker Exports
- Verification Summaries (`summary.json`)

## Continuous Integration
This repo uses GitHub Actions for Linting, Testing, and SonarQube Analysis.
