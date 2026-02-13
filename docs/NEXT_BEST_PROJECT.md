# Cerberus Alpha Evolution - NEXT_BEST_PROJECT.md

The current system relies on basic OHLCV indicators (RSI, EMA, VWAP) which provide low signal-to-noise ratios. To reach "Institutional" or "Prop" level performance, we need to shift focus to Order Flow, statistical stationarity, and probabilistic decision-making.

## High-Conviction Projects

### 1. Order Flow Imbalance (Microstructure Alpha)
Stop looking at *price* as the lead indicator. Price is the *result* of volume imbalance.
- **Goal**: Implement **Trade Flow Imbalance (TFI)** in the `FeaturePipeline`.
- **Logic**: Use tick-level data (or high-resolution trades) to categorize volume as aggressive buying (at/above ask) or aggressive selling (at/below bid).

### 2. The Net-Gamma "Pin" (Flow Alpha)
Market makers are forced to hedge based on dealer gamma exposure. This creates predictable "magnets" or "walls" for price.
- **Goal**: Integrate **Net GEX (Gamma Exposure)** using the `Unusual Whales` API.
- **Logic**: Identify the "Zero Gamma" level and use strike-level concentration to estimate where price is likely to consolidate.

### 3. Fractional Differentiation (Statistical Alpha)
Standard price returns (pct_change) destroy the "memory" of a price series. We need stationarity (mean = 0, var = 1) *without* losing the long-term trend information.
- **Goal**: Implement **Fractional Differentiation** for all technical indicators.
- **Utility**: Significantly improves the training stability of any future ML models.

### 4. Meta-Labeling (Risk Alpha)
Instead of hard-coded "if/then" rules, use a model to predict the *conviction* of a signal.
- **Goal**: Create a **Meta-Solver** that vets signals.
- **Mechanism**: Strategy generates a "Signal" -> Meta-Solver predicts "Probability of Win" -> We only execute if Prob > 0.65.

## Roadmap for Next Sprints
1. **Sprint A**: TFI implementation in `FeatureCalculator`.
2. **Sprint B**: GEX integration via UW.
3. **Sprint C**: Data logging for Meta-Labeling training.
