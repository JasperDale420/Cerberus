# Phase 1 Remaining: Zero-Cost Algorithm Upgrades (#6-9)

**Date:** 2026-03-12
**Scope:** 4 independent algorithm upgrades from ADVANCED_STRATEGY_RESEARCH.md Phase 1
**Prerequisites:** Top 5 upgrades already implemented (BOCPD, Permutation Entropy, Causal Filter, VPIN, CPPI)

---

## Overview

Four remaining zero-cost upgrades that require no new data sources. All are config-gated with backward-compatible fallbacks.

---

## Module 1: Intraday Momentum Strategy (§4.2)

**Research basis:** Gao et al. (2018) "Intraday Momentum" + Gao & Suss (2024) cross-asset validation. First-half-hour return predicts last-half-hour return. Validated across 45 years and 62 markets.

### Signal Logic

```
r_morning = cumulative return 9:30-10:00 ET
r_first_half = cumulative return 9:30-12:30 ET

Entry condition (at ~14:30 ET):
  sign(r_morning) == sign(r_first_half)
  AND abs(r_first_half) > min_move_threshold (default 0.3%)
  AND confluence score >= threshold

Direction: same as r_first_half sign
Exit: MOC or 15:50 hard stop
```

### Architecture

**New file:** `src/strategies/intraday_momentum.py`
- Inherits from `BaseStrategy`
- Tracks per-symbol morning/midday returns via bar accumulation
- Time window: 14:30-15:50 ET only (power hour entries)
- Stores first-half data in `_symbol_sessions: Dict[str, SessionData]` keyed by date

**Confluence scoring (6 factors):**

| Factor | Weight | Logic |
|--------|--------|-------|
| Morning-midday agreement | 0.30 | Both positive or both negative |
| First-half magnitude | 0.25 | Larger moves = stronger signal |
| Volume confirmation | 0.15 | Above-average volume in first half |
| Regime alignment | 0.15 | Trend regime matches direction |
| ADX strength | 0.10 | Trending market (ADX > 20) |
| Session quality | 0.05 | Power hour > late close |

**Stop/target:**
- Stop: 1.5x ATR(14) from entry, structure-adjusted
- Target: VWAP or 1.5R minimum (same pattern as MeanReversionPro)
- Hard exit: 15:50 ET (10 min before close)

**Config params:**
- `min_move_threshold`: 0.003 (0.3% first-half move required)
- `confluence_threshold`: 60.0
- `time_window_start`: "14:30"
- `time_window_end`: "15:50"
- `max_hold_minutes`: 80

### Test plan
- Morning return tracking accumulation
- Signal generation when morning/midday agree
- No signal when they disagree
- No signal before 14:30 or after 15:50
- Session reset on new trading day
- Confluence scoring thresholds

---

## Module 2: Anti-Fragile Strategy Classification (§6.5)

**Research basis:** Taleb (2012) "Antifragile" + Schwalbach & Auret (2025) formalized barbell approach. Some strategies are convex (benefit from volatility); current system reduces ALL uniformly.

### Strategy Classifications

| Convexity Class | Strategies | Behavior in Stress |
|----------------|-----------|-------------------|
| `convex` | Trend Rider Pro, Momentum strategies | Increase size in HIGH/SHOCK vol |
| `linear` | ORB, Flow Alpha, Intraday Momentum | Maintain size (use global defaults) |
| `concave` | Mean Reversion Pro, Gap strategies | Reduce size more aggressively |

### Per-Class Regime Multiplier Overrides

```python
ANTIFRAGILE_OVERRIDES = {
    "convex": {
        "vol": {"low": 0.80, "normal": 1.00, "high": 1.30, "shock": 1.50},
        "risk": {"risk_on": 1.00, "neutral": 1.00, "risk_off": 1.20},
    },
    "concave": {
        "vol": {"low": 1.10, "normal": 1.00, "high": 0.40, "shock": 0.00},
        "risk": {"risk_on": 1.00, "neutral": 0.80, "risk_off": 0.30},
    },
    # "linear" uses global defaults (no overrides)
}
```

### Architecture

**Modified files:**
1. `src/config/models.py` — Add `ConvexityClass` enum (`convex`, `linear`, `concave`) and `antifragile_overrides` dict to `RiskConfig`
2. `src/engine/risk.py` — Modify `_get_regime_multiplier()` to accept optional `strategy_name` param and apply per-class overrides

**Integration in risk.py:**
```python
def _get_regime_multiplier(self, market_state, strategy_name=None) -> float:
    # ... existing axis multiplication ...
    # NEW: apply antifragile overrides if strategy has a convexity class
    if strategy_name:
        convexity = self._get_convexity_class(strategy_name)
        if convexity != "linear":
            combined = self._apply_antifragile_overrides(combined, snapshot, convexity)
    return combined
```

**Backward compatibility:** Default convexity_class is `"linear"` (no change from current behavior). Only strategies explicitly classified get different treatment.

### Test plan
- Convex strategy gets higher multiplier in SHOCK vol
- Concave strategy gets lower multiplier in SHOCK vol
- Linear strategy unchanged (uses global defaults)
- Unknown strategy defaults to linear
- Config parsing and validation

---

## Module 3: Ornstein-Uhlenbeck Dynamic Thresholds (§3.2)

**Research basis:** Leung & Li (2015) "Optimal Mean Reversion Trading" + Tang (2018) regime-switching OU. Replace fixed entry thresholds with estimated mean-reversion speed.

### OU Parameter Estimation

**Discrete MLE for theta (mean-reversion speed):**
```
Given VWAP distance series {x_1, ..., x_n}:
  a = exp(-theta * dt)  where dt = bar interval
  theta_hat = -ln(Σ(x_t - x_bar)(x_{t-1} - x_bar) / Σ(x_{t-1} - x_bar)²) / dt
  mu_hat = mean of series (long-run mean, should be ~0 for VWAP distance)
  sigma_hat = std(x_t - a * x_{t-1}) / sqrt((1 - a²) / (2 * theta))
  half_life = ln(2) / theta_hat
```

### Dynamic Threshold Logic

```python
# Compute scaling factor based on current vs median theta
theta_ratio = median_theta / current_theta  # >1 when slow reversion, <1 when fast
scaling = clamp(theta_ratio, 0.5, 2.0)

dynamic_vwap_threshold = base_vwap_threshold * scaling
dynamic_bb_threshold = base_bb_threshold * scaling
```

**Intuition:** When mean reversion is fast (high theta), enter closer to the mean (tighter thresholds). When slow (low theta), require wider deviations before entering.

### Architecture

**New file:** `src/analysis/ou_estimator.py`
- `OUResult` dataclass: `theta`, `mu`, `sigma`, `half_life`, `scaling_factor`
- `OUEstimator.__init__(lookback=60, min_observations=30, dt=1/390)`
- `OUEstimator.update(vwap_distance: float) -> Optional[OUResult]`
- Rolling window MLE, returns None until `min_observations` reached
- Tracks `median_theta` over 1200-bar z-score window for scaling reference

**Modified file:** `src/strategies/mean_reversion_pro.py`
- Add `self._ou: Dict[str, OUEstimator] = {}` in `__init__`
- In `on_bar()`, after computing `vwap_dist`, update OU estimator
- Pass dynamic thresholds to `_detect_side()` (add optional params)
- Config: `ou_enabled: bool = False`, `ou_lookback: int = 60`
- Fallback: static thresholds when OU disabled or insufficient data

### Test plan
- Theta estimation from synthetic OU process
- Half-life calculation correctness
- Dynamic threshold scaling (fast theta = tighter, slow = wider)
- Clamp bounds respected (0.5x-2.0x)
- Fallback to static when insufficient data
- Integration with _detect_side

---

## Module 4: Variance Risk Premium Signal (§5.2)

**Research basis:** Bollerslev, Tauchen & Zhou (2009) "Expected Stock Returns and Variance Risk Premia". VRP = IV² - RV² is the single strongest predictor of future equity returns in the academic literature. Sharpe ~0.8 as standalone signal.

### VRP Computation

```
IV_proxy = (VXX_price / 10)²   # VXX ≈ sqrt(30d IV) × 10
RV = realized_vol²              # From existing EWMA in regime.py

VRP = IV_proxy - RV

Normalized: vrp_zscore = (VRP - rolling_mean(VRP)) / rolling_std(VRP)

Classification:
  vrp_zscore > 1.0  → RISK_ON  (high premium = market compensating for fear)
  vrp_zscore < -1.0 → RISK_OFF (low premium = complacency or realized catching up)
  else              → NEUTRAL
```

### Architecture

**New file:** `src/analysis/vrp.py`
- `VRPResult` dataclass: `vrp_raw`, `vrp_zscore`, `iv_proxy`, `realized_vol`
- `VRPCalculator.__init__(window=60, zscore_window=1200)`
- `VRPCalculator.update(vxx_price: float, realized_vol: float) -> Optional[VRPResult]`
- Rolling z-score normalization over 1200 bars

**Modified file:** `src/analysis/regime.py`
- Add `from src.analysis.vrp import VRPCalculator`
- Add `self._vrp = VRPCalculator(logger=logger)` in `__init__`
- In `_classify_risk()`: when VRP calculator has data, use VRP z-score classification; else fall back to existing VXX momentum thresholds
- Add `vrp_score` field to `MarketRegimeSnapshot` (optional float, default 0.0)
- Config: `vrp_enabled: bool = True` (on by default since it enhances existing VXX data)

### Test plan
- VRP computation from known VXX + RV values
- Z-score normalization
- RISK_ON classification when VRP high
- RISK_OFF classification when VRP low
- Fallback to VXX momentum when insufficient data
- Edge cases: zero vol, constant VXX

---

## Shared File Modifications

| File | Module 1 | Module 2 | Module 3 | Module 4 |
|------|----------|----------|----------|----------|
| `config/models.py` | - | ConvexityClass enum + antifragile config | - | - |
| `engine/risk.py` | - | _get_regime_multiplier strategy override | - | - |
| `strategies/mean_reversion_pro.py` | - | - | OU estimator integration | - |
| `analysis/regime.py` | - | - | - | VRP integration |
| `core/domain.py` | - | - | - | `vrp_score` field on snapshot |

**No overlapping shared files between agents** — each can integrate directly.

---

## Swarm Execution Plan

4 parallel agents, each creating standalone module + tests + integrating into their specific shared files.

| Agent | New Files | Modified Files | Test Count |
|-------|-----------|---------------|------------|
| 1: Intraday Momentum | `strategies/intraday_momentum.py`, `tests/unit/test_intraday_momentum_unit.py` | None | ~10 |
| 2: Anti-Fragile | (config only) `tests/unit/test_antifragile_unit.py` | `config/models.py`, `engine/risk.py` | ~8 |
| 3: OU Estimator | `analysis/ou_estimator.py`, `tests/unit/test_ou_estimator_unit.py` | `strategies/mean_reversion_pro.py` | ~8 |
| 4: VRP Signal | `analysis/vrp.py`, `tests/unit/test_vrp_unit.py` | `analysis/regime.py`, `core/domain.py` | ~8 |

Post-swarm: run full test suite to verify no regressions.
