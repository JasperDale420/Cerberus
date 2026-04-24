# Cerberus Autoresearch

You are a quant researcher. **Your explicit goal: build a strategy whose cumulative OOS return beats SPY buy-and-hold by at least 2x over the same time span.** The composite score is the optimization metric; the `ratio_vs_spy` in the benchmark line is the pass/fail check. If composite score is climbing but `ratio_vs_spy < 2.0`, the strategy is not good enough yet — keep iterating.

## Evaluation setup (what you're optimizing against)

- **Data window:** 2016-06-01 → 2026-03-19 (full available bar history, ~10 years).
- **Walk-forward:** rolling 12-month train → 6-month OOS, 3-month final holdout. ~18 OOS windows.
- **Universe:** SPY, QQQ, AAPL, NVDA, TSLA, AMD, AMZN, META (8 symbols).
- **Scoring:** composite of Sortino/PF/Calmar per OOS window × regime-diversity multiplier × param-stability × LOC penalty. `AUTORESEARCH_RESULT` is the main line; `AUTORESEARCH_BENCHMARK` shows `strategy_return_pct` vs `spy_return_pct` and the `ratio_vs_spy` (goal ≥ 2.00). Use both. The composite score is what's being maximized; the SPY ratio is the acceptance gate.
- **Anti-overfit guards wired in:** randomized WFO splits, param-stability CV penalty (CV > 0.3 penalizes), regime-diversity multiplier (penalizes single-regime concentration), 3-month final holdout you never see during iteration.

## The Loop

You are spawned by a driver script. Each iteration:

1. Read the **Last Result** in your prompt — it has scores, trade analysis, and regime breakdown
2. Read `src/strategies/<name>.py` — understand the current code
3. Make **ONE** change to improve the score
4. `ruff check src/strategies/<name>.py` — catch errors before wasting 45 min
5. `git commit` with a descriptive message (commit directly on `main` — see branching rule below)
6. **STOP.** The driver runs the evaluation. You never run it.

## Branching rule

**Stay on `main`. Do NOT create branches or worktrees.** Every iteration is an atomic commit on `main`; failed experiments are reverted with `git reset --hard HEAD~1`. Never run `git checkout -b`, `git switch -c`, or `git worktree add`.

## Simplicity Criterion

All else being equal, **simpler is better**.

- A 0.1 score improvement that adds 20 lines of code? Not worth it.
- Removing code and getting equal results? Definitely keep — that's a simplification win.
- A strategy with 3 clean factors scoring 3.0 beats one with 8 noisy factors scoring 3.5.
- The composite score includes a LOC penalty. Under 50 lines gets a bonus. Over 100 gets penalized.

When in doubt, **delete**. Every line of code is a liability unless it demonstrably improves the score.

## When You're Stuck

If your last 3 changes were all discarded, **stop tweaking and think**:

1. Re-read the trade analysis — which exit reasons dominate?
   - `stop_loss` > 40% → stops are too tight, widen them
   - `max_hold` > 30% → targets are unreachable, lower them
   - `target` < 20% → targets are too loose, tighten them
2. Check BUY vs SELL PnL — if one side is profitable, go single-direction
3. Check session phase — if morning is profitable but midday loses, restrict time window
4. Check regime windows — which regimes score > 0? Double down on those
5. Try combining the best elements from your last 2-3 near-misses
6. Try something radically different — new indicator, different entry logic, opposite hypothesis

**Never repeat a failed approach.** Check the iteration history in your prompt.

## Crash Handling

- Typo or missing import? Fix and re-commit (counts as same iteration).
- 2 consecutive crashes on the same error? The approach is broken — skip it, try something else.
- Import fails after `git reset`? The previous iteration's fix was reverted. Re-apply the fix AND make your change in one commit.

## Open Sandbox (files you CAN modify)

- `src/strategies/*.py` — strategy files. Create new or modify existing.
- `config/strategies.yaml` — params and activation policies

## Frozen Files (NEVER modify)

Everything else. The evaluation, backtest, engine, and data pipeline are frozen.

## Creating a New Strategy

```python
from src.strategies.base import BaseStrategy
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState

class MyStrategy(BaseStrategy):
    name = "my_strategy"  # must match config key in strategies.yaml

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.threshold = float(config.get("threshold", 0.5))

    def on_bar(self, symbol, bar, symbol_state, market_state):
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, 20):
            return None

        # Your signal logic: return Signal or None
        close = bar.close
        # ... compute indicators from symbol_state.bars or symbol_state.indicators
        # Use self._create_signal(symbol, bar, side, stop, target, meta) to build Signal
        return None
```

Add to `config/strategies.yaml`:
```yaml
my_strategy:
  enabled: true
  threshold: 0.5
  activation:
    session: [opening, midday, power_hour]
    trend: [up, down, flat]
    vol: [low, normal, high]
    liquidity: [good, thin]
    risk: [risk_on, neutral, risk_off]
    min_confidence: 0.0
```

**IMPORTANT:** Keep activation permissive. Restrictive activation = 0 trades = wasted iteration.

## Available Data

**Bar:** `bar.open`, `bar.high`, `bar.low`, `bar.close`, `bar.volume`, `bar.vwap`, `bar.time`

**SymbolState:**
- `symbol_state.bars` — deque of recent bars (use `list(symbol_state.bars)` for array ops)
- `symbol_state.indicators` — dict of precomputed EMA, RSI, ATR, BB
- `symbol_state.meta["regime_labels"]` — `{regime_trend, regime_vol, liquidity_regime, near_earnings, near_fomc, opex_week}`
- `symbol_state.position` — current position (None if flat)

**Signal:** `self._create_signal(symbol, bar, side, stop_price, target_price, meta={})`
- `side`: `OrderSide.BUY` or `OrderSide.SELL`
- `stop_price`: absolute price for stop loss
- `target_price`: absolute price for take profit
