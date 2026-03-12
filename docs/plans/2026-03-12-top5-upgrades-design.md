# Top 5 Zero-Cost Cerberus Upgrades — Design Document

**Date:** 2026-03-12
**Status:** Approved
**Scope:** 5 enhancements requiring no new data sources, implemented as parallel workstreams

---

## 1. BOCPD Changepoint Detection

**File:** `src/analysis/bocpd.py` (new)
**Modifies:** `src/analysis/regime.py`, `src/core/domain.py`

Bayesian Online Changepoint Detection replaces 5-bar majority-vote hysteresis. Provides probabilistic regime transition timing.

**Math:** Adams & MacKay run-length posterior with Normal-Inverse-Gamma conjugate prior:
```
P(r_t | x_{1:t}) ∝ P(x_t | r_t, x_{t-r_t:t-1}) × P(r_t | r_{t-1}) × P(r_{t-1} | x_{1:t-1})
```
Hazard function `H(r) = 1/λ` (geometric prior on segment length). O(1) per bar with truncated run-length (max 200).

**Interface:**
- `BOCPDDetector.__init__(hazard_lambda, prior_mu, prior_kappa, prior_alpha, prior_beta)`
- `BOCPDDetector.update(value) -> BOCPDResult(changepoint_prob, run_length_posterior, map_run_length)`
- `MarketRegimeSnapshot` gains: `changepoint_probability: float`, `bars_since_changepoint: int`

**Integration:** `MarketContextService.update()` feeds log returns to BOCPD. High changepoint probability (>0.7) triggers regime confidence reduction.

## 2. Permutation Entropy Regime Overlay

**File:** `src/analysis/entropy.py` (new)
**Modifies:** `src/core/domain.py`, `src/config/models.py`

Rolling Permutation Entropy (PE) and Lempel-Ziv Complexity (LZC). Low complexity = structured (predictable move coming). High complexity = noise.

**Math:**
```
PE(m, τ) = -Σ_π P(π) × ln(P(π))
```
Embedding dimension m=5, delay τ=1, 60-bar window. Normalized to [0, 1] where 1 = maximum entropy (random).

LZC: Count distinct subsequences in binary-encoded return series (up=1, down=0). Normalized by theoretical maximum.

**Interface:**
- `EntropyAnalyzer.__init__(pe_order=5, pe_delay=1, window=60)`
- `EntropyAnalyzer.update(returns_value) -> EntropyResult(pe_normalized, lzc_normalized, complexity_zscore)`
- New enum `ComplexityRegime`: `STRUCTURED`, `NORMAL`, `RANDOM`
- `MarketRegimeSnapshot` gains: `complexity: ComplexityRegime`, `entropy_score: float`

**Integration:** Z-score < -2 = STRUCTURED (big move imminent), Z-score > +2 = RANDOM (noise). Feeds `regime_risk_multipliers["complexity"]` in risk config.

## 3. Causal Inference Signal Filter

**File:** `src/analysis/causal_filter.py` (new)
**Modifies:** `src/engine/strategy_engine.py`

Tests whether strategy signals have causal (not just associational) relationship to future returns.

**Math:** Granger causality with lag selection via BIC:
```
r_{t+h} = α + Σ_k β_k × r_{t-k} + Σ_k γ_k × signal_{t-k} + ε_t
F-test: H0: all γ_k = 0
```
With confounding control: include market return, sector return, volatility as covariates.

**Interface:**
- `CausalSignalFilter.__init__(min_observations=500, max_lag=10, significance=0.05)`
- `CausalSignalFilter.update_scores(strategy_trade_history)` — monthly re-estimation
- `CausalSignalFilter.get_causal_strength(strategy_name) -> float` — returns 0-1 score
- `CausalSignalFilter.should_allow(strategy_name, threshold=0.3) -> bool`

**Integration:** `StrategyEngine._evaluate_strategies()` calls `causal_filter.should_allow()` before forwarding signals. Strategies below threshold get signals suppressed. Logs causal scores for monitoring. Starts permissive (all pass) until enough data accumulates.

## 4. VPIN Toxicity Filter

**File:** `src/analysis/vpin.py` (new)
**Modifies:** `src/strategies/mean_reversion_pro.py` (and other MR strategies)

Volume-Synchronized Probability of Informed Trading. High VPIN = informed flow = mean reversion will fail.

**Math:**
```
VPIN = Σ_n |V_buy_n - V_sell_n| / (N × V_bucket)
```
Using Bulk Volume Classification (BVC): buy fraction = Φ((close - open) / σ) where Φ is standard normal CDF. Volume bucket size = daily volume / N (typically N=50).

**Interface:**
- `VPINCalculator.__init__(n_buckets=50, window_buckets=50)`
- `VPINCalculator.update(bar: Bar) -> VPINResult(vpin: float, buy_volume: float, sell_volume: float)`
- VPIN range: [0, 1]. Above 0.7 = high toxicity.

**Integration:** Mean reversion strategies call `vpin.update()` and check `vpin < threshold` before generating signals. Added as pre-entry gate in `mean_reversion_pro.py`. Signal metadata gets `vpin_score` field.

## 5. CPPI Drawdown-Controlled Sizing

**File:** `src/engine/cppi.py` (new)
**Modifies:** `src/engine/risk.py`, `src/config/models.py`

Replaces binary circuit breaker with Constant Proportion Portfolio Insurance. Continuous position size reduction as drawdown accumulates.

**Math:**
```
floor_t = max(floor_{t-1}, portfolio_t × (1 - max_drawdown))
cushion_t = portfolio_t - floor_t
exposure_t = multiplier × cushion_t
cppi_fraction = min(1.0, exposure_t / portfolio_t)
```
Multiplier typically 3-5. Floor ratchets up with equity highs.

**Interface:**
- `CPPISizer.__init__(config: CPPIConfig)`
- `CPPISizer.update_equity(equity: float)` — called on each PnL update
- `CPPISizer.get_cppi_multiplier() -> float` — returns 0.0-1.0 fraction
- `CPPIConfig(enabled, multiplier, max_drawdown, min_exposure)` added to `RiskConfig`

**Integration:** `RiskManager._calculate_qty()` applies `cppi_multiplier` after regime multiplier. Replaces the binary `risk_mode == "reduced"` with continuous scaling. Daily loss circuit breaker remains as a hard backstop.

---

## Conflict Resolution

Agents 1 (BOCPD) and 2 (Entropy) both need to extend `MarketRegimeSnapshot` in `domain.py` and add logic to `regime.py`. Agent 5 (CPPI) and Agent 2 (Entropy) both extend `config/models.py`.

**Strategy:** Each agent creates only their standalone module + unit tests. A final integration pass by the lead merges changes into shared files (`regime.py`, `domain.py`, `config/models.py`, `strategy_engine.py`).

## Test Strategy

Each agent writes unit tests in `tests/unit/test_<module>_unit.py`. Integration tests written during merge pass.
