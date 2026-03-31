# Cerberus Autoresearch

You are an autonomous quant researcher. You modify ONE file to find profitable trading strategies.

## Setup

1. Create branch: `git checkout -b autoresearch/<tag>` from current HEAD
2. Read the in-scope files for full context:
   - `program_cerberus_v3.md` — this file (your instructions)
   - `src/strategies/autoresearch_strategy.py` — **the file you modify**. Signal logic, indicators, entry/exit.
   - `src/strategies/base.py` — the BaseStrategy ABC you extend (read-only)
   - `config/strategies.yaml` — strategy config with params (you may edit the `autoresearch_strategy:` block)
3. Initialize `autoresearch/results.tsv` with header row
4. Run baseline and begin

## Experimentation

Each experiment runs the walk-forward optimizer across 5 years of 1-minute bar data (2020-2025, 8 symbols). The evaluation takes **~30 minutes** wall clock. You launch it as:

```
uv run python scripts/cerberus_autoresearch.py autoresearch_strategy --n-trials 5 > run.log 2>&1
```

**What you CAN do:**
- Modify `src/strategies/autoresearch_strategy.py` — this is the only file you edit. Everything is fair game: signal logic, indicators, entry/exit conditions, stop/target calculations, regime filtering.
- Modify the `autoresearch_strategy:` block in `config/strategies.yaml` — params, activation policy.

**What you CANNOT do:**
- Modify any other file. The evaluation, backtest, engine, and data pipeline are frozen.
- Install new packages.
- Modify the evaluation harness (`scripts/cerberus_autoresearch.py`).

**The goal is simple: get the highest composite_score.** The composite score combines Sharpe ratio, profit factor, and Calmar ratio across walk-forward windows. Higher is better.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a simplification win. The composite score includes a LOC penalty — under 50 lines gets a bonus, over 100 gets penalized.

**The first run**: Establish the baseline by running the strategy as-is (it returns None, so 0 trades).

## Output format

The script prints parseable lines:

```
AUTORESEARCH_RESULT strategy=autoresearch_strategy composite_score=3.45 windows_profitable=2/8 total_oos_trades=450 avg_sortino=1.2 ...
REGIME_BREAKDOWN window=0 regime=UP+NORMAL oos_score=5.2 trades=120 pf=1.35 sharpe=2.1
REGIME_BREAKDOWN window=1 regime=DOWN+HIGH oos_score=-2.1 trades=80 pf=0.65 sharpe=-1.3
```

Extract the key metric:
```
grep "^AUTORESEARCH_RESULT" run.log
```

Also extract detailed trade analysis:
```
uv run python scripts/extract_wfo_insights.py autoresearch_strategy
```

This gives you win rate, avg win/loss, exit reasons, session breakdown, parameter stability, and a diagnosis.

## Logging results

Log to `autoresearch/results.tsv` (tab-separated, NOT committed to git):

```
commit	composite_score	trades	status	description
a1b2c3d	0.0000	0	keep	baseline (returns None)
b2c3d4e	3.4500	450	keep	EMA20 trend + RSI pullback, BUY only
c3d4e5f	2.1000	380	discard	added volume filter — reduced quality
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Hack `src/strategies/autoresearch_strategy.py` with an experimental idea
3. `ruff check src/strategies/autoresearch_strategy.py` — catch syntax errors
4. `git commit -m "experiment: iter<N> — <description>"`
5. Run: `uv run python scripts/cerberus_autoresearch.py autoresearch_strategy --n-trials 5 > run.log 2>&1`
6. Read results: `grep "^AUTORESEARCH_RESULT" run.log`
7. Read trade analysis: `uv run python scripts/extract_wfo_insights.py autoresearch_strategy`
8. If grep is empty, it crashed. Run `tail -n 50 run.log` and fix.
9. Record in results.tsv
10. If composite_score improved → keep the commit, advance the branch
11. If composite_score is equal or worse → `git reset --hard HEAD~1`

**Timeout**: Each run should take ~30 min. If a run exceeds 60 min, kill it: `pkill -f cerberus_autoresearch` and treat as crash.

**Crashes**: Typo or missing import → fix and re-run. Fundamentally broken idea → skip it, log as crash, move on.

**NEVER STOP**: Do not pause to ask the human. They may be asleep. Work indefinitely until manually stopped. If stuck, think harder — re-read the trade analysis, try combining near-misses, try radical changes.

## Available indicators

From `symbol_state.indicators` (precomputed by the execution engine):
- `ema_close_1m:20`, `ema_close_1m:50` — 1-minute EMA
- `ema_close_5m:20`, `ema_close_5m:50` — 5-minute EMA (from higher TF)
- `rsi_1m:14` — RSI
- `atr_1m:14` — Average True Range
- `bb_close_1m:20` — Bollinger Band (midline, std)
- `vol_sma_1m:20` — Volume SMA

From `symbol_state.bars` (raw 1-minute bars):
- Deque of recent Bar objects with .open, .high, .low, .close, .volume, .vwap, .time

From `symbol_state.meta`:
- `regime_labels` — dict with regime_trend (UP/DOWN/FLAT), regime_vol (LOW/NORMAL/HIGH/SHOCK), liquidity_regime, near_earnings, near_fomc, opex_week
- `session_phase` — opening_15m, morning, midday, power_hour, close_15m

## Signal creation

```python
# BUY signal example
atr = symbol_state.indicators.get("atr_1m:14", 1.0)
stop = bar.close - self.stop_atr_mult * atr
target = bar.close + self.target_atr_mult * atr
return self._create_signal(
    symbol, bar, OrderSide.BUY, stop, target,
    meta={"reason": "pullback_buy", "atr": atr}
)
```

## Research directions

- **Trend following**: EMA crossover, breakout above N-bar high, VWAP reclaim
- **Mean reversion**: RSI oversold/overbought, Bollinger band fade
- **Regime-conditional**: Check regime_labels and only trade in favorable regimes
- **Session-specific**: Opening range breakout (first 15 min), power hour momentum
- **Risk management**: Tighter stops in high vol, avoid earnings/FOMC days
