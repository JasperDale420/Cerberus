---
name: strategy-status
description: Show current status of all Cerberus strategies — enabled/disabled, WFO results, optimization history, and readiness for live trading
---

# Strategy Status

Quick overview of all Cerberus strategies and their current state.

## Usage

```
/strategy-status [strategy_name]
```

Without arguments, shows all strategies. With a name, shows detailed info for that strategy.

## Workflow

### 1. Read Strategy Registry

Check which strategies exist and are enabled:

```bash
# Active strategies from config
grep -A2 "enabled:" config/strategies.yaml

# V2 strategies (current generation)
grep "class.*BaseStrategy" src/strategies/*.py
```

### 2. Check Optimization History

For each strategy, check:

```bash
# WFO results
ls artifacts/optimization/*{strategy}* 2>/dev/null

# Auto-tuned params
grep -A20 "{strategy}" config/strategies.auto.yaml 2>/dev/null

# Recent git activity
git log --oneline -5 -- src/strategies/{strategy}.py
```

### 3. Check Param Spaces

Verify the strategy has a defined optimization param space:

```bash
grep -A20 "def suggest.*{strategy}" src/analytics/param_spaces.py
```

### 4. Output Table

```
CERBERUS STRATEGY STATUS
========================

Strategy              Enabled  Last WFO     +Windows  Param Stability  Trades  Status
--------              -------  --------     --------  ---------------  ------  ------
mean_reversion_pro    yes      2024-03-15   4/6       3/5 stable       1200    Optimized
trend_rider_pro       yes      2024-03-20   3/6       4/7 stable       884     In Progress
flow_alpha            yes      never        -         -                -       Not Optimized
orb_v2                yes      2024-02-10   1/6       2/5 stable       200     Needs Work
pair_trading_v2       yes      2024-03-18   5/8       6/7 stable       3400    Ready
rsi_bounce            yes      2024-03-26   3/6       4/7 stable       916     In Progress
momentum_fade         no       never        -         -                -       Disabled

LEGEND:
  Ready        = >60% positive OOS windows, stable params, >100 trades/window
  In Progress  = Has WFO data but doesn't meet Ready criteria
  Needs Work   = WFO attempted but poor results
  Not Optimized = No WFO run yet
  Disabled     = Turned off in config
```

### 5. Detailed View (single strategy)

When a specific strategy is requested, also show:
- Current YAML config params
- Auto-tuned overrides (from strategies.auto.yaml)
- Last 10 git commits touching that strategy
- WFO per-window breakdown (scores, trades, Sharpe)
- Param stability details (mean, CV per param)
- Activation policy (which regimes it trades in)

## Key Files

- `config/strategies.yaml` — Base strategy configs
- `config/strategies.auto.yaml` — Agent-tuned overrides
- `src/strategies/` — Strategy implementations
- `src/analytics/param_spaces.py` — Optimization param spaces
- `artifacts/optimization/` — WFO result JSONs
- `src/engine/strategy_engine.py` — Activation policies
