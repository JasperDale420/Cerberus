# Regime-Aware Multi-Strategy Portfolio System

**Date:** 2026-03-20
**Status:** Approved
**Problem:** Single-strategy optimization overfits to one market regime. TRP v4 (optimized on 2025) scores -500 OOS on 2024 WFO windows.

## Motivation

Markets are non-stationary. Any fixed parameter set decays as regimes shift. The current approach — optimize one strategy on one time period — is a dead end. We need:

1. **Diversification** across complementary strategies
2. **Scoring that penalizes regime-specific overfitting**
3. **Dynamic capital allocation** that adapts to what's working

## Architecture Overview

Two phases, built sequentially:

```
Phase B: Multi-Strategy Portfolio (diversification baseline)
  └─ Enable all 7 V2 strategies
  └─ Enable HRP cross-strategy allocation
  └─ Establish per-strategy and portfolio metrics across 2020-2026

Phase A: Cross-Validated Autoresearch (anti-overfit scoring)
  └─ Replace single-period scoring with multi-window cross-validation
  └─ Minimum viability gate (breakeven in every window)
  └─ Regime-conditional parameter sets (future layer)
```

## Phase B: Multi-Strategy Portfolio

### Goal
Run all 7 V2 strategies concurrently with HRP allocation and measure the portfolio baseline across 6 years of data (2020-2026).

### Changes

**1. New config: `config/backtest_portfolio.yaml`**
- Enables all 7 V2 strategies with existing parameters
- Enables HRP allocator (60-day lookback, 5-40% weight bounds, rebalance every 5 days)
- Disables all V1 legacy strategies
- Uses full data range

**2. Strategies enabled (no parameter changes):**

| Strategy | Style | Expected Regime Strength |
|----------|-------|-------------------------|
| trend_rider_pro | Trend following | Trending, low-normal vol |
| mean_reversion_pro | Mean reversion | Flat/choppy, normal-high vol |
| flow_alpha | Flow-driven | Any trend, good liquidity |
| orb_v2 | Breakout | Opening session, trending |
| rsi_bounce | Oversold/overbought | Any, high vol spikes |
| momentum_fade | Fade momentum spikes | Any, VWAP deviation |
| pair_trading_v2 | Stat-arb pairs | Flat, mean-reverting |

**3. HRP allocation:**
- `HRPAllocator` with default config (already implemented in `src/engine/hrp.py`)
- Correlated strategies get less capital; uncorrelated get more
- Minimum 3 active strategies required to engage HRP

**4. Baseline metrics to collect:**
- Per-strategy: Sharpe, PF, win rate, trade count, max DD — per year
- Portfolio: Combined Sharpe, max DD, Calmar, correlation matrix
- Regime breakdown: strategy performance by trend/vol axis

### What stays the same
- All existing activation policies (regime gating per strategy)
- HMM gate configs
- Risk management (daily loss limits, position caps)
- Individual strategy logic

## Phase A: Cross-Validated Autoresearch

### Goal
Replace single-period scoring with multi-window cross-validation so autoresearch cannot overfit to one regime.

### Changes

**1. Multi-window scoring in `autoresearch_score.py`:**

Replace single 2025 window with 5 non-overlapping windows:
- Window 1: 2020-06 → 2021-06 (COVID recovery, bull run)
- Window 2: 2021-06 → 2022-06 (peak → bear market)
- Window 3: 2022-06 → 2023-06 (bear → recovery)
- Window 4: 2023-06 → 2024-06 (AI bull)
- Window 5: 2024-06 → 2025-06 (mixed/choppy)

**2. Scoring formula:**
- Run backtest on each window independently
- Per-window composite score uses existing formula (PnL/Sharpe/PF/WR/trade count)
- Final score = `mean(window_scores) - 1.0 * std(window_scores)`
  - Rewards consistency, penalizes regime-dependent performance
  - A strategy that scores +0.5 everywhere beats one that scores +2.0 in one window and -0.5 in others

**3. Minimum viability gate:**
- Strategy must score >= 0.0 (breakeven) in at least 4 of 5 windows
- If raw logic can't survive multiple regimes, it needs redesign, not tuning
- Gate checked before optimization begins

**4. Regime-conditional parameter sets (future layer):**
- 2-3 parameter sets per strategy, activated by HMM regime state
- E.g., TRP uses tighter stops in HIGH vol, wider targets in LOW vol
- Implemented after portfolio baseline and cross-validated scoring are working

## Data Requirements

- 68 symbols, 1-minute bars, 2020-01-01 → 2026-03-20 (download in progress)
- ~6 years of data enables robust cross-validation windows
- Some newer tickers (HOOD, RIVN, IONQ, etc.) won't have pre-IPO data — expected

## Success Criteria

1. **Portfolio Sharpe > individual strategy Sharpe** across full period
2. **Max drawdown reduced** by diversification vs single-strategy
3. **Cross-validated scores stable** (CV < 0.5 across windows)
4. **No single strategy dominates** capital allocation persistently (HRP working)
5. **Autoresearch iterations improve all windows** simultaneously, not just one

## Risks

- Some V2 strategies may not have been maintained since being disabled — may need debugging
- HRP needs sufficient trade history to compute correlations — cold start problem
- 6 years of 1-min data for 68 symbols = large backtests (~30min+ per run)
- Pair trading requires cointegration pairs to be defined and may need separate data

---

# Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable multi-strategy portfolio with HRP allocation, then upgrade autoresearch scoring to cross-validate across regime windows.

**Architecture:** Create a new portfolio backtest config enabling 5 backtestable V2 strategies (flow_alpha and pair_trading_v2 excluded — no historical data). Enable HRP allocator via risk config. Create a portfolio analysis script for per-strategy/per-year metrics. Then upgrade autoresearch scoring to evaluate across 5 non-overlapping 1-year windows with a consistency-penalizing composite formula.

**Tech Stack:** Python, PyArrow/Parquet, Optuna, asyncio, YAML config

---

## Phase B: Multi-Strategy Portfolio

### Task 1: Create portfolio backtest config

**Files:**
- Create: `config/backtest_portfolio.yaml`

**Step 1: Create the portfolio config file**

Copy `config/backtest_v2.yaml` as a base, then make these changes:
- Enable `mean_reversion_pro`, `orb_v2`, `rsi_bounce`, `momentum_fade` (set `enabled: true`)
- Keep `trend_rider_pro` enabled with current iter60 params
- Keep `flow_alpha: enabled: false` (no historical flow data for backtest)
- Keep `pair_trading_v2: enabled: false` (needs cointegration pair data)
- Keep `disable_flow_strategies: true`
- Add HRP config under `risk:`:
  ```yaml
  risk:
    hrp:
      enabled: true
      lookback_days: 60
      min_strategies: 3
      rebalance_interval: 5
      min_weight: 0.05
      max_weight: 0.40
  ```
- Increase `max_open_positions: 12` and `max_positions_per_strategy: 3` (5 strategies need room)
- Increase `max_trades_per_day: 80` (5 strategies generate more signals)
- Expand universe to include all 68 symbols from `offline_symbols.txt`

**Step 2: Verify config loads without errors**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run python -c "from src.core.config import ConfigLoader; c = ConfigLoader(); cfg = c.load_config('config/backtest_portfolio.yaml'); print('Strategies:', [k for k,v in cfg.get('strategies',{}).items() if v.get('enabled')]); print('HRP:', cfg.get('risk',{}).get('hrp',{}))"`

Expected: Lists 5 enabled strategies and HRP config.

**Step 3: Commit**

```bash
git add config/backtest_portfolio.yaml
git commit -m "feat: add multi-strategy portfolio backtest config with HRP"
```

### Task 2: Run baseline portfolio backtest (2024-2025)

**Files:**
- Create: `scripts/run_portfolio_backtest.py`

**Step 1: Create the portfolio backtest runner**

```python
#!/usr/bin/env python
"""Run multi-strategy portfolio backtest across date ranges."""

import asyncio
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["EMPIRE_LOG_LEVEL"] = "WARNING"
os.environ.setdefault("EMPIRE_LOG_FORMAT", "json")

os.chdir("/Users/jacobmcmillan/Empire/Cerberus")
sys.path.insert(0, ".")

from src.backtest.runner import run_backtest  # noqa: E402


async def main():
    windows = [
        ("2020-06-01", "2021-06-01", "COVID recovery → bull"),
        ("2021-06-01", "2022-06-01", "Peak → bear market"),
        ("2022-06-01", "2023-06-01", "Bear → recovery"),
        ("2023-06-01", "2024-06-01", "AI bull"),
        ("2024-06-01", "2025-06-01", "Mixed/choppy"),
        ("2025-06-01", "2026-03-20", "Recent"),
    ]

    config = "config/backtest_portfolio.yaml"
    data_dir = "data/bars_2023_2025"

    results = {}
    for start, end, label in windows:
        print(f"\n{'='*60}")
        print(f"  Window: {start} → {end} ({label})")
        print(f"{'='*60}")

        report = await run_backtest(start, end, config, data_dir=data_dir)
        if report is None:
            print(f"  FAILED — no report")
            results[label] = {"error": "backtest_failed"}
            continue

        metrics = report.to_dict()
        results[label] = {
            "start": start,
            "end": end,
            "net_pnl": round(metrics.get("net_pnl", 0), 2),
            "sharpe": round(metrics.get("sharpe_ratio", 0), 3),
            "profit_factor": round(metrics.get("profit_factor", 0), 2),
            "winrate": round(metrics.get("winrate", 0) * 100, 1),
            "n_trades": metrics.get("n_trades", 0),
            "max_drawdown_pct": round(metrics.get("max_drawdown_pct", 0), 2),
            "calmar": round(metrics.get("calmar_ratio", 0), 3),
        }
        print(f"  PnL: ${metrics.get('net_pnl',0):,.2f}  Sharpe: {metrics.get('sharpe_ratio',0):.3f}  Trades: {metrics.get('n_trades',0)}")

    out_dir = "artifacts/portfolio"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/portfolio_baseline_2020_2026.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to {out_path}")
    print(f"{'='*60}")
    for label, m in results.items():
        if "error" not in m:
            print(f"  {label:30s}  PnL=${m['net_pnl']:>10,.2f}  Sharpe={m['sharpe']:>6.3f}  Trades={m['n_trades']:>4d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Run the backtest (start with one window to verify)**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run python scripts/run_portfolio_backtest.py`

Expected: Backtest completes for each window, results saved to `artifacts/portfolio/portfolio_baseline_2020_2026.json`.

**Step 3: Commit**

```bash
git add scripts/run_portfolio_backtest.py
git commit -m "feat: add multi-strategy portfolio backtest runner with per-window analysis"
```

### Task 3: Analyze results and diagnose strategy issues

After the baseline backtest runs, analyze:
- Which strategies generated trades? (some may be broken after being disabled)
- Per-strategy Sharpe across windows — which are additive?
- Portfolio Sharpe vs TRP-only Sharpe — is diversification helping?
- HRP weight evolution — is allocation adapting?

This is a manual analysis step. If any strategies fail to generate signals, debug them before proceeding to Phase A.

---

## Phase A: Cross-Validated Autoresearch Scoring

### Task 4: Create cross-validated scoring script

**Files:**
- Create: `scripts/autoresearch_score_cv.py`
- Keep: `scripts/autoresearch_score.py` (unchanged, for backwards compat)

**Step 1: Create the cross-validated scoring script**

```python
"""Cross-validated autoresearch scoring across multiple regime windows.

Scores strategy performance across 5 non-overlapping 1-year windows.
Final score penalizes inconsistency: mean(scores) - 1.0 * std(scores).
"""

import asyncio
import json
import math
import sys

from src.backtest.runner import run_backtest

WINDOWS = [
    ("2020-06-01", "2021-06-01"),
    ("2021-06-01", "2022-06-01"),
    ("2022-06-01", "2023-06-01"),
    ("2023-06-01", "2024-06-01"),
    ("2024-06-01", "2025-06-01"),
]

CONFIG = "config/backtest_portfolio.yaml"
DATA_DIR = "data/bars_2023_2025"

MIN_PASSING_WINDOWS = 4  # Must score >= 0 in at least 4 of 5 windows


def compute_composite_score(metrics: dict) -> float:
    """Same scoring formula as autoresearch_score.py for consistency."""
    pnl = metrics.get("net_pnl", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    pf = metrics.get("profit_factor", 0)
    winrate = metrics.get("winrate", 0)
    n_trades = metrics.get("n_trades", 0)

    pnl_score = pnl / 10000.0
    sharpe_score = sharpe / 2.0
    pf_score = max(0, pf - 1.0)
    wr_score = (winrate - 0.50) / 0.15

    if n_trades < 50:
        trade_score = -1.0
    elif n_trades < 200:
        trade_score = (n_trades - 50) / 150.0 * 0.5
    elif n_trades <= 800:
        trade_score = 0.5 + 0.5 * (1.0 - abs(n_trades - 400) / 400.0)
    elif n_trades <= 2000:
        trade_score = max(0, 0.5 - (n_trades - 800) / 2400.0)
    else:
        trade_score = -0.5

    composite = 0.30 * pnl_score + 0.25 * sharpe_score + 0.20 * pf_score + 0.15 * wr_score + 0.10 * trade_score
    return round(composite, 4)


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    window_scores = []
    window_details = []

    for start, end in WINDOWS:
        report = await run_backtest(start, end, args.config, data_dir=args.data_dir)
        if report is None:
            window_scores.append(-999)
            window_details.append({"window": f"{start}→{end}", "error": "backtest_failed"})
            continue

        metrics = report.to_dict()
        score = compute_composite_score(metrics)
        window_scores.append(score)
        window_details.append({
            "window": f"{start}→{end}",
            "score": score,
            "net_pnl": round(metrics.get("net_pnl", 0), 2),
            "sharpe": round(metrics.get("sharpe_ratio", 0), 3),
            "n_trades": metrics.get("n_trades", 0),
        })

    # Cross-validated composite: penalize inconsistency
    valid_scores = [s for s in window_scores if s > -999]
    if len(valid_scores) < 3:
        cv_score = -999
    else:
        mean_score = sum(valid_scores) / len(valid_scores)
        std_score = math.sqrt(sum((s - mean_score) ** 2 for s in valid_scores) / len(valid_scores))
        cv_score = round(mean_score - 1.0 * std_score, 4)

    # Viability gate: must pass in MIN_PASSING_WINDOWS windows
    passing_windows = sum(1 for s in valid_scores if s >= 0.0)
    passed_gate = passing_windows >= MIN_PASSING_WINDOWS

    result = {
        "autoresearch_score": cv_score,
        "mean_score": round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else -999,
        "std_score": round(math.sqrt(sum((s - sum(valid_scores)/len(valid_scores))**2 for s in valid_scores) / len(valid_scores)), 4) if valid_scores else 0,
        "passing_windows": passing_windows,
        "viability_gate": passed_gate,
        "window_scores": window_scores,
        "window_details": window_details,
    }

    print(json.dumps(result), file=sys.stderr)
    print(f"AUTORESEARCH_SCORE={cv_score}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Run to verify it works**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run python scripts/autoresearch_score_cv.py`

Expected: Runs 5 backtests, prints cross-validated score.

**Step 3: Commit**

```bash
git add scripts/autoresearch_score_cv.py
git commit -m "feat: add cross-validated autoresearch scoring across 5 regime windows"
```

### Task 5: Update autoresearch config to use CV scoring

**Files:**
- Modify: `scripts/autoresearch_score.py` — add `--cv` flag that switches to multi-window mode

**Step 1: Add --cv flag to existing autoresearch_score.py**

Add an `--cv` argument. When set, import and delegate to the CV script's `main()`.

**Step 2: Verify backward compatibility**

Run without `--cv`: should behave exactly as before (single 2025 window).
Run with `--cv`: should run 5-window cross-validation.

**Step 3: Commit**

```bash
git add scripts/autoresearch_score.py
git commit -m "feat: add --cv flag to autoresearch scoring for cross-validated mode"
```

### Task 6: Update offline_symbols.txt and data directory references

**Files:**
- Modify: `config/offline_symbols.txt` — ensure all 68 symbols are listed
- Verify: `data/bars_2023_2025/` has parquet files for all symbols once download completes

**Step 1: Verify symbol list is complete**

Compare `offline_symbols.txt` with downloaded data files.

**Step 2: Data health check**

Run per-symbol: min date, max date, row count, check for gaps > 3 trading days.

**Step 3: Commit any fixes**

```bash
git add config/offline_symbols.txt
git commit -m "chore: update offline symbol list to full 68-symbol universe"
```
