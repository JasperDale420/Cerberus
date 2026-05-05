# Cerberus Autoresearch

You are a quant researcher. **Your explicit goal: build a strategy whose cumulative OOS return beats SPY buy-and-hold by at least 3x over the same time span.** The composite score is the optimization metric; the `ratio_vs_spy` in the benchmark line is the pass/fail check. If composite score is climbing but `ratio_vs_spy < 3.0`, the strategy is not good enough yet — keep iterating.

The 3.0 target is set above the "obvious" 2.0 because of capital-gains tax friction (see "After-tax math" below). 2x pretax with 100% short-term gains barely matches a buy-and-hold investor after tax; 3x pretax produces real alpha after tax.

## Evaluation setup (what you're optimizing against)

- **Data window:** 2016-06-01 → 2026-03-19 (full available bar history, ~10 years).
- **Walk-forward:** rolling 12-month train → 6-month OOS, with the final 3 months reserved as a holdout (2026-01-01 → 2026-03-19). ~18 OOS windows. The holdout is **now validated** — see "Holdout validation" below.
- **Universe:** 16 symbols, regime-diverse — indices (SPY, QQQ, IWM), high-beta tech (AAPL, NVDA, TSLA, AMD, AMZN, META), defensives (KO, JNJ, XLP, XLU), bond proxy (TLT), commodity (GLD), cyclical financials (XLF). Bear and flat-regime specialists previously had no instruments to express edge against; the prior 8-symbol mega-cap-tech list overweighted bull regimes.
- **Composite score = `ratio_vs_spy` (with adjustments).** This is THE metric. Both strategy returns and SPY are compounded as growth factors over the scoring windows, so the comparison is apples-to-apples. The benchmark uses three sign-safe modes:
  - **bull** (SPY > +1%): `ratio_vs_spy = strategy_return_pct / spy_return_pct` — direct "Nx SPY return" goal.
  - **bear** (SPY < −1%): `ratio_vs_spy = 1 + alpha/|spy|` — alpha=0 means matched SPY (1.0); breaking even when SPY lost 10% is alpha=10% on |spy|=10% → 2.0 (2x SPY in bear-market sense).
  - **flat** (|SPY| ≤ 1%): `ratio_vs_spy = strategy_return_pct` — absolute return as alpha proxy when the benchmark is noise; needed to keep FLAT-regime specialists evaluable.
  - **Goal: composite ≥ 3.0** in any mode.
- **Hard gates** — if any of these fail, score is forced to **−2.0** regardless of any other metric:
  - `windows_profitable_pct ≥ 40%` — at least 40% of scoring windows must have **positive net PnL** (not just "didn't trip a hard-reject sentinel"). This is the lie-prevention check.
  - `ratio_vs_spy > 0` — strategy must be net profitable over the span (in flat mode this means absolute return > 0; in bear mode it means alpha > −|spy|).
  - `total_oos_trades ≥ 5 × n_windows` — minimum activity floor.
  - When `--target-regime` is set: at least one OOS window must classify as that regime (else gate fails with `regime_filter_no_match=<regime>`).
- **Adjustments after gates pass:** `composite = ratio_vs_spy * loc_multiplier`, where `loc_multiplier = 1 + loc_penalty * 0.05` (LOC penalty/bonus is multiplicative, ±10% effective range). If `cv_max > 0.3`, multiply by `max(0.5, 1 − (cv_max − 0.3))`.
- **Anti-overfit guards:** randomized WFO splits, param-stability CV multiplier, regime-diversity scoring inside the WFO, plus a post-KEEP holdout backtest on 2026-Q1.

The old "average of profitable-only windows" scoring was gameable — a strategy losing money in 12/13 windows could still score high from the one lucky window. The new score collapses to **−2.0** in that scenario.

## Holdout validation

After every KEEP, the driver re-runs the kept iteration on the reserved 2026-01-01 → 2026-03-19 window using WFO-selected consensus params from `artifacts/autoresearch/<strategy>_latest.json`. The holdout emits its own `HOLDOUT_RESULT` line with `ratio_vs_spy` for the held-out window. Verdict rules:

- **HOLDOUT FAIL → downgrade to discard** if (a) OOS `ratio_vs_spy > 0` AND (b) holdout `ratio_vs_spy < 0` (sign-flip) OR holdout `ratio_vs_spy < 0.5 × OOS ratio`. The kept commit is rolled back; `BEST_COMMIT` reverts to its prior value.
- **HOLDOUT OK** otherwise — the iter is committed as the new baseline.

You see the holdout line in the next iteration's prompt. If holdout fails, the trade analysis from the WFO is still useful — but the param surface that earned the OOS score does not generalize. Try a structurally different change next iter, not a parameter tweak.

## After-tax math

The 3.0 pretax target is calibrated so that genuine alpha survives short-term capital-gains tax friction:

- Strategy avg hold time ≪ 1 year ⇒ all gains are short-term, taxed as ordinary income. Modal high-earner federal+state ≈ **35%**.
- SPY buy-and-hold over 10 years ⇒ all gains are long-term, ≈ **18%** combined (15% LTCG federal + state).
- After-tax math (bull mode): `after_tax_ratio = (strat_pct × 0.65) / (spy_pct × 0.82)`.
- A 3.0 pretax ratio with these constants gives `after_tax_ratio ≈ 2.4` — genuine 2.4× SPY *after* tax. A 2.0 pretax ratio gives `after_tax_ratio ≈ 1.6` — barely worthwhile.

The harness emits `after_tax_ratio` and `after_tax_mode` alongside `ratio_vs_spy` in `AUTORESEARCH_BENCHMARK`. **It is informational, not gated** — the gate stays on pretax `composite_score`. But you should glance at `after_tax_ratio` before declaring victory: a strategy with `ratio_vs_spy = 3.0` but `after_tax_ratio < 2.0` is still suspect.

If you can find ways to lengthen avg hold (e.g., overnight holds during clear trends) without losing edge, the after-tax tail wins disproportionately because the strategy moves toward the 18% LTCG bucket. **Trade-frequency reduction is a free after-tax score boost.**

## Regime data policy

**You MUST use `market_state.regime_snapshot` (Labeler B, live `MarketContextService`) for any regime gating.** Available axes: `trend` (UP/DOWN/FLAT), `vol` (LOW/NORMAL/HIGH/SHOCK), `liquidity` (GOOD/THIN/STRESSED), `risk` (RISK_ON/NEUTRAL/RISK_OFF), `session` (PREMARKET/OPENING/MIDDAY/POWER_HOUR/CLOSE).

**Do NOT read `symbol_state.meta["regime_labels"]`.** That dict is populated only in backtests (by `src/backtest/runner.py` from `data/regime_labeled_v2/`). Nothing in the live engine writes it. A strategy that filters on `regime_labels.get("regime_vol")` will get `{}` in production, default to `"NORMAL"`, and run with **no filter** — train/live divergence is silent and inverts the risk profile.

This rule was discovered the hard way: see [reason/260501-0159-regime-labels-critique/p0_audit_memo.md](reason/260501-0159-regime-labels-critique/p0_audit_memo.md). The four phase strategies (`regime_trend_up`, `regime_bear`, `regime_flat`, `regime_adaptive`) all already use `regime_snapshot`. Keep it that way.

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
