---
name: wfo
description: Run walk-forward optimization for a Cerberus strategy using the optuna_harness framework
---

# Walk-Forward Optimization

Run a standardized WFO for any Cerberus strategy.

## Usage

```
/wfo <strategy_name> [options]
```

## Arguments

- `strategy_name` (required): Strategy to optimize (e.g., `rsi_bounce`, `trend_rider_pro`, `mean_reversion_pro`)

Options (passed inline):
- `dataset`: Which bar data to use — `bars_2024` (1yr/37sym), `bars_2023_2025` (6yr/68sym), `bars_5yr` (5yr/36sym). Default: `bars_2023_2025`
- `symbols`: Number of symbols or explicit list. Default: 20 (diversified cross-sector)
- `train_months`: Training window size. Default: 6
- `test_months`: OOS window size. Default: 2
- `trials`: Optuna trials per window. Default: 30
- `workers`: Parallel workers. Default: 2

## Workflow

1. **Validate** the strategy exists in `src/strategies/` and has a param space in `src/analytics/param_spaces.py`
2. **Load** the base config from `config/backtest_v2.yaml`
3. **Configure** the WFO using `src/analytics/optuna_harness.WalkForwardOptimizer`
4. **Run** the optimization with the specified parameters
5. **Save** results to `artifacts/optimization/{strategy}_wfo_hardening.json`
6. **Report** summary: OOS scores, positive windows, param stability, trade counts
7. **Validate** results using the `backtest-validator` agent if available

## Implementation

Generate and run a script based on `scripts/run_wfo_rsi_bounce.py` as a template. Key imports:

```python
from src.analytics.optuna_harness import WalkForwardOptimizer
```

The script must:
- Suppress logging (`logging.disable(logging.CRITICAL)`)
- Use `config/backtest_v2.yaml` as base config
- Enable only the target strategy, disable all others
- Use `sqlite://` as database_url (in-memory, no contention)
- Print param stability analysis at the end
- Print a METRIC line for autoresearch parsing

## Default Symbol Universe (20 symbols, cross-sector)

```python
["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "META", "MSFT", "GOOGL",
 "JPM", "GS", "BAC", "XOM", "CVX", "NFLX", "UBER", "COIN", "PLTR", "SOFI"]
```

## Available Datasets

| Dataset | Symbols | Date Range | Size |
|---------|---------|------------|------|
| `data/bars_2024` | 37 | 2024-01-02 to 2024-12-31 | 325 MB |
| `data/bars_5yr` | 36 | 2020-01-02 to 2024-12-31 | 525 MB |
| `data/bars_2023_2025` | 68 | 2020-01-01 to 2026-03-19 | 1.6 GB |

## Adversarial Review (Required)

Before presenting results as a conclusion, this analysis must get an adversarial review per the monorepo's Data Analysis Review policy (CLAUDE.md): either an Opus subagent explicitly instructed to challenge the methodology (overfitting, look-ahead/leakage, cherry-picked windows, unsupported claims), or `/codex:adversarial-review` (or the `codex` skill) run with the strongest available GPT model (currently `gpt-5.5`) at high/xhigh reasoning effort. Report the review's findings alongside the results, not as a separate follow-up.
