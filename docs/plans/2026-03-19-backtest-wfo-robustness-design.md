# Backtest & WFO Robustness Upgrade — Design Document

**Date:** 2026-03-19
**Goal:** Make the backtest and WFO systems more realistic, analytically rich, and actionable for strategy improvement.
**Approach:** Layered architecture — new capabilities composed via clean interfaces on top of the existing working engine.

---

## 1. Pluggable Fill Model

### Problem
Fixed `slippage_bps` applies identical slippage regardless of order size or bar volume.

### Design

**Protocol:**
```python
# src/backtest/fill_models/protocol.py
class FillModel(Protocol):
    def compute_fill(self, order: Order, bar: Bar, account: Account) -> FillResult: ...

@dataclass
class FillResult:
    fill_price: float
    filled_qty: int
    commission: float
    slippage_bps: float
    market_impact: float
```

**Implementations:**
1. `FixedSlippageFillModel` — current behavior, preserved for backward compat.
2. `VolumeAwareFillModel` (new default):
   - `participation_rate = order_qty / bar_volume`
   - `effective_slippage = base_bps + (participation_rate * impact_coefficient)`
   - Default: ~10bps at 5% participation, configurable `impact_coefficient`.
   - Tiered commission (per-share with volume breaks).

**Integration:**
- `SimulatedOrderExecutor` receives `FillModel` at construction.
- `process_bar()` delegates fill calculation to the model.
- Config selects model:

```yaml
backtest:
  fill_model: volume_aware
  fill_model_params:
    base_slippage_bps: 2.0
    impact_coefficient: 200
    commission_per_share: 0.001
```

---

## 2. Per-Strategy Overnight Position Handling

### Problem
`force_flat_at_1600` is a global kill switch that liquidates good trend trades.

### Design

**Strategy config:**
```yaml
strategies:
  trend_rider_pro:
    allow_overnight: true
    max_hold_days: 5
    overnight_stop_mult: 1.5
  orb_v2:
    allow_overnight: false
```

**Base class additions:**
- `allow_overnight: bool = False`
- `max_hold_days: int = 0` (0 = unlimited)
- `overnight_stop_mult: float = 1.0`

**Runner logic (replaces global flatten):**
At 15:55 ET for each open position:
- If `strategy.allow_overnight is False` → flatten.
- Else if `overnight_stop_mult > 1.0` → widen existing stop by multiplier.
- Else if `max_hold_days > 0` and `position.hold_days >= max_hold_days` → flatten.

**Overnight gap handling:** Existing stop-fill logic already fills at `bar.open` when price gaps through stop — no changes needed.

**Position tracking:** `hold_days` computed as trading days between `entry_date` and current bar date.

---

## 3. Monte Carlo & Statistical Validation

### Problem
A single equity curve is one sample path. No confidence intervals on metrics.

### Design

**Module:** `src/analytics/monte_carlo.py`

**Bootstrap Monte Carlo:**
- Input: list of trade PnLs from backtest.
- Resample with replacement N times (default 10,000).
- Per resample: build equity curve, compute Sharpe, max DD, final equity, CAGR.
- Output: percentile bands (5th, 25th, 50th, 75th, 95th) for each metric.

```python
@dataclass
class MonteCarloResult:
    n_simulations: int
    metric_distributions: dict[str, PercentileBands]
    probability_of_loss: float
    probability_of_ruin: float        # DD > ruin_threshold
    worst_case_drawdown: float        # 5th percentile max DD
    confidence_interval_95: tuple[float, float]
```

**Integration:**
- Auto-runs at end of `run_backtest()` when enabled.
- Callable standalone from WFO for per-window analysis.
- Results in `BacktestReportCard.monte_carlo`.
- Deflated Sharpe exposed in standard backtest report (not just WFO).

```yaml
analytics:
  monte_carlo:
    enabled: true
    n_simulations: 10000
    ruin_threshold_pct: 30.0
    confidence_level: 0.95
```

---

## 4. Data Quality & Benchmark Comparison

### Problem
Bad data silently corrupts results. No benchmark comparison to distinguish alpha from beta.

### Data Quality Checks

**Module:** `src/backtest/data_quality.py` — runs before bar replay.

Checks:
- **Gap detection:** Missing bars within RTH (>1 gap). Flag symbols >5% missing.
- **Outlier detection:** `|close/prev_close - 1| > 15%` flagged (possible splits/bad data).
- **Volume anomalies:** Zero volume during RTH, or >10x rolling 20-bar average.
- **Staleness:** Consecutive identical close prices.
- **Coverage summary:** Per-symbol total/expected bars, coverage %, flagged issues.

Symbols <80% coverage get a warning. <50% get excluded.

### Benchmark Comparison

Added to `BacktestReportCard`:

```python
@dataclass
class BenchmarkComparison:
    benchmark_symbol: str
    benchmark_return_pct: float
    strategy_alpha: float
    strategy_beta: float
    information_ratio: float
    up_capture: float
    down_capture: float
```

Uses `index_symbol` (SPY) daily returns already loaded.

```yaml
analytics:
  benchmark:
    symbol: SPY
    include_metrics: [alpha, beta, information_ratio, capture_ratios]
  data_quality:
    min_coverage_pct: 80.0
    exclude_below_pct: 50.0
    max_gap_bars: 5
```

---

## 5. WFO Enhancements — Automated Holdout & Parameter Sensitivity

### Automated Holdout Validation

After all walk-forward windows, auto-run best OOS params on holdout:

```python
@dataclass
class HoldoutResult:
    params_used: dict[str, Any]
    holdout_sharpe: float
    holdout_pf: float
    holdout_max_dd: float
    holdout_n_trades: int
    holdout_score: float
    oos_to_holdout_ratio: float
    passed: bool                      # ratio > 0.4
```

Param selection: window with best OOS score (not IS).

### Parameter Sensitivity Analysis

**Module:** `src/analytics/param_sensitivity.py`

From Optuna study data (`study.trials_dataframe()`):
- Per-param Spearman rank correlation with objective → rank by influence.
- Top-3 most influential param pairs: 2D interpolated surfaces for dashboard heatmaps.

```python
@dataclass
class SensitivityResult:
    param_name: str
    values: list[float]
    scores: list[float]
    correlation: float
    sensitivity_rank: int
    pairwise_surfaces: dict[str, np.ndarray]
```

```yaml
analytics:
  holdout:
    auto_validate: true
    pass_threshold: 0.4
  param_sensitivity:
    enabled: true
    top_pairs: 3
```

---

## 6. Analytics Dashboard (EmpireUI)

### API Endpoints (Cerberus side)

```
GET /api/backtest/runs                        → list of completed runs
GET /api/backtest/runs/{run_id}/equity         → daily equity + benchmark
GET /api/backtest/runs/{run_id}/trades         → trade list with regime tags
GET /api/backtest/runs/{run_id}/monte-carlo    → percentile bands
GET /api/backtest/runs/{run_id}/regime-splits  → pre-aggregated regime breakdowns
GET /api/wfo/runs/{run_id}/sensitivity         → param sensitivity data
```

**Data persistence:** JSON files in `results/` directory, keyed by run hash.

### View 1: Equity Curve Overlays
- Multi-line chart: strategy equity curves + SPY benchmark.
- WFO periods shaded: in-sample (blue), out-of-sample (green), holdout (yellow).
- Toggle strategies on/off.
- Drawdown subplot (underwater chart) below.
- Monte Carlo 5th/95th bands as shaded region.

### View 2: Trade Scatter Plots
- Scatter: hold duration (X) vs PnL (Y), colored by strategy.
- Scatter: entry time-of-day (X) vs PnL (Y), colored by win/loss.
- Filter controls: strategy, date range, PnL range.
- Click dot → trade detail popover.

### View 3: Regime Performance Breakdown
- Grouped bar charts: strategy returns by regime axis (trend/vol/session).
- Heatmap: strategies (rows) x regime states (columns), cell = avg PnL.
- Highlights where each strategy thrives vs bleeds.

---

## 7. Post-Backtest Diagnostics Engine

### Problem
Need systematic analysis that identifies where performance leaks and what to try next.

### Design

**Module:** `src/analytics/diagnostics.py` — runs after every backtest.

**Analyses:**
1. **Strategy contribution** — rank by net PnL, Sharpe, marginal diversification. Flag negative contributors.
2. **Regime mismatch detection** — cross-reference activation policies with per-regime performance. Flag strategies allowed in regimes where they consistently lose.
3. **Time-of-day edge decay** — bucket PnL by 30-min entry windows. Identify when edge is strongest. Recommend tightening session activation.
4. **Holding period analysis** — PnL by exit type (target/stop/time-limit/flatten). Suggest `max_hold_minutes` adjustments.
5. **Risk-adjusted sizing** — high Sharpe + low allocation = opportunity. Low Sharpe + high allocation = risk.
6. **Consecutive loss streaks** — flag max losers > 2x expected for win rate. Indicates regime clustering.

```python
@dataclass
class DiagnosticsReport:
    strategy_rankings: list[StrategyRanking]
    regime_mismatches: list[RegimeMismatch]
    time_edge_map: dict[str, list[TimeSlotEdge]]
    hold_analysis: dict[str, HoldAnalysis]
    sizing_suggestions: list[SizingSuggestion]
    loss_streak_flags: list[LossStreakFlag]
    summary: str   # Plain-text top 3-5 findings (human + LLM readable)
```

```yaml
analytics:
  diagnostics:
    enabled: true
    min_trades_for_analysis: 20
```

---

## File Structure (New/Modified)

```
src/backtest/
  fill_models/
    __init__.py
    protocol.py              # FillModel protocol + FillResult
    fixed.py                 # FixedSlippageFillModel
    volume_aware.py          # VolumeAwareFillModel
  data_quality.py            # Pre-backtest data checks
  executor.py                # Modified: delegate to FillModel
  runner.py                  # Modified: per-strategy overnight, analytics hooks

src/analytics/
  monte_carlo.py             # Bootstrap MC simulation
  param_sensitivity.py       # WFO parameter sensitivity
  diagnostics.py             # Post-backtest diagnostics engine
  optuna_harness.py          # Modified: automated holdout validation

src/strategies/
  base.py                    # Modified: overnight fields

src/backtest/
  backtest_report.py         # Modified: benchmark, monte carlo, diagnostics

src/api/                     # New or extended
  backtest_api.py            # REST endpoints for dashboard

EmpireUI/
  src/pages/backtest/
    EquityCurveOverlay.tsx
    TradeScatterPlot.tsx
    RegimeBreakdown.tsx
```
