# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **Comprehensive analytics report card**: Ported 11 analysis functions from trading-bot covering MAE/MFE, statistical tests, Monte Carlo, PnL distribution, risk metrics (Omega/Ulcer/VaR/CVaR), rolling metrics, drawdown catalog, trade clustering, cost sensitivity, time breakdowns, and entry/exit efficiency. Adapted for Cerberus dict-based trades with configurable R-multiple PnL keys and equity minute-bar conventions (98,280 bars/year).

- **Analytics integrated into BacktestReportCard**: Every backtest now auto-generates advanced analytics (Omega ratio, Ulcer Index, VaR/CVaR, Monte Carlo ruin probability, bootstrap CI, statistical tests). Results included in `to_dict()` output and markdown reports under "Advanced Analytics" section.

- **Analytics integrated into WFO harness**: Walk-forward optimization saves per-window analytics reports to `artifacts/optimization/runs/{strategy}/{run_tag}/reports/` for post-run diagnostics.

### Fixed

- **MetaLabeler missing symmetric GEX filter for shorts**: The GEX heuristic only blocked longs in deeply negative GEX but did not block shorts in deeply positive GEX (where dealer hedging pins price up). Added the symmetric check and extracted the threshold into a named constant.

- **CVaR sizer division by zero**: `_calc_multiplier()` could crash with `ZeroDivisionError` if `max_acceptable_cvar` was configured as 0. Added zero guard on both the sizing calculation and the threshold-exceeded logging.

- **Silent swallow of invalid strategy activation configs**: `build_activation_policies_from_config()` caught all exceptions with bare `pass`, hiding malformed YAML. Now logs warnings with `exc_info` for both invalid configs and unrecognized regime enum values.

- **reconcile_loop crashes silently**: If `reconcile_broker_state()` threw an unhandled exception, the entire reconciliation task died with no logging or restart. Added try/except with error logging inside the loop body.

- **Empty equity curve crashes backtest report**: `_compute_equity_metrics()` accessed `equities[0]` without checking if the list was empty. Added early return guard.

- **GARCH volatility log of non-positive prices**: `volatility.py` passed raw prices to `np.log()` without filtering zero/negative values, producing `RuntimeWarning: invalid value encountered in log`. Now filters non-positive prices before computing log returns.

- **Stream dispatch silently swallows callback errors**: `_dispatch_event()` in `data/client.py` caught all callback exceptions with bare `pass`. Bar/trade/quote processing failures were completely invisible. Now logs errors with `exc_info`.

- **Silent fail-open in time window checks**: `in_trading_window()` and `in_time_window_str()` returned `True` on exceptions with no logging. Timezone failures were invisible to operators. Added warning-level logging.

- **Granger causality test failure silent**: `flow_alpha.py` swallowed Granger test exceptions with bare `pass`. Added debug-level logging for visibility.

- **flow_alpha GARCH normalization direction**: `flow_zscore * garch_cond_vol` nearly zeroed out the flow signal (cond_vol ≈ 0.015). Changed to divide-based normalization (`flow_zscore / (cond_vol * 200)`) matching GARCH standardized residuals convention and aligning with the `/3.0` fallback at typical volatility.

- **momentum_fade velocity division by zero**: `_compute_exhaustion()` divided by `closes_list[-1 - velocity_lookback]` without a zero guard. Added early return if base price is zero.

- **Market state meta update silent failure**: `MarketStateManager.update()` swallowed exceptions when writing `trend_score` and `regime_tags` to `state.meta` with bare `pass`. Added warning-level logging.

- **Risk mode set failure silent**: `set_risk_mode()` swallowed exceptions with bare `pass`. Since this is called to halt trading on risk breaches, a silent failure could allow trading to continue when it should be stopped. Added error-level logging.

- **DB buffer metrics failure silent**: `_check_db_trading_halt()` swallowed exceptions when reading `write_buffer_len()`/`write_buffer_max()`, leaving metrics at 0. This prevented the trading halt check from triggering during actual DB degradation. Added warning-level logging.

- **Feature snapshot persist_feature_snapshot() called with positional args**: `pipeline.py` called `persist_feature_snapshot(feat, now)` but the function signature uses keyword-only args (`*, features, as_of_ts`). This would raise `TypeError` at runtime. Fixed to use keyword arguments.

- **gex_data uninitialized on fetch failure**: If `fetch_flow()` or `fetch_gex()` raised an exception, `gex_data` was never assigned but referenced later, causing `NameError`. Added `gex_data = []` to the exception handler.

- **Scan removal loop crashes on missing symbol**: `_process_scan_removals()` accessed `self.symbol_states[sym]` directly without checking if the symbol exists, causing `KeyError` and breaking cleanup of remaining symbols. Added existence check.

- **Fill data parse failure silent**: When broker fill events couldn't be parsed to float (qty/price), the fill was silently dropped with no logging, causing local position tracking to diverge from broker state. Added warning-level logging.

- **Scheduler hardcodes `--mode live`**: `CerberusScheduler._run_daily_session()` spawned the trading subprocess with `--mode live` regardless of config. Changed to read mode from config with `paper` as default, matching the safety invariant that paper mode is always the default.

- **Scheduler uses raw structlog instead of central logger**: `scheduler.py` imported `structlog.get_logger()` directly instead of using `src.core.logger.get_logger()`, bypassing the central logging configuration (JSON formatting, file rotation).

- **Hard stop time parse failure silent**: `BaseStrategy.is_past_hard_stop()` silently returned `False` if the configured hard_stop_time was malformed, allowing strategies to trade past their safety cutoff. Added warning-level logging on parse failure.

- **Monte Carlo simulation crashes on empty trade list**: `run_monte_carlo()` accessed `equity_curve[-1]` without checking for empty input, crashing with `IndexError` when backtest produces 0 trades. Added early return with neutral result.

- **Holding period calculation failure silent**: `PositionManager._build_closed_trade()` silently swallowed errors computing holding period, making it impossible to diagnose timezone/type mismatches. Added debug-level logging.

- **CVaR GPD formula invalid for tail index >= 1.0**: The GPD-based CVaR formula divides by `(1 - xi)`, which flips sign when `xi >= 1`, producing positive CVaR (nonsensical "expected profit in tail"). The GPD mean is undefined for `xi >= 1`. Added guard to reject GPD results in this case and fall back to empirical CVaR.

- **NoopOrderExecutor.cancel_all_for_symbol() returns None instead of int**: All other executor implementations (`OrderExecutor`, `GatewayOrderExecutor`, `BacktestExecutor`) return `int`, but `NoopOrderExecutor` returned `None`. Fixed to return `0` for LSP compliance.

- **Mean reversion GARCH z-score vs raw % threshold mismatch**: `generate_signal()` converted VWAP distance to a GARCH conditional z-score (sigma units) then compared it against `vwap_dist_threshold` (raw %, default 0.003). Any z-score above 0.003 sigma trivially passes, effectively disabling the VWAP distance gate when GARCH is active. Fixed to use raw VWAP distance for direction detection; GARCH influence flows through the adaptive threshold engine instead.

- **Position reconciliation destroys exit management state**: `_reconcile_single_position()` created a new Position object preserving only 8 fields, losing stop_price, trailing stop state, partial exit tracking, and open_risk. After reconciliation, positions briefly had no stop protection and trailing stops reset to current price instead of historical high water mark. Fixed to mutate the existing position in-place when the side matches, preserving all exit management state.

- **BOCPD posterior underflow permanently kills changepoint detection**: If numerical underflow caused `evidence` to reach zero during Bayesian Online Changepoint Detection, the posterior distribution remained all-zeros permanently — BOCPD could never recover. Added reset to changepoint prior when evidence drops below 1e-300.

- **Kelly non-robust mode overwrites raw fraction for logging**: In the non-robust path, `raw_kelly` was overwritten with `max_equity_pct` or `min_equity_pct` for "logging" purposes, but the overwritten value fed into the actual fractional Kelly scaling. This caused all-winners to get a *smaller* position fraction than near-all-winners (non-monotonic). Fixed to use a separate variable for logging.

- **HRP same-day trades create duplicate date entries**: When multiple trades completed on the same day for the same strategy, `record_daily_return()` appended separate entries instead of accumulating into a single daily return. This corrupted the return matrix alignment used for correlation computation. Fixed to sum same-day PnL.

- **Trend regime classifies zero cumulative return as DOWN**: `_classify_trend()` used `UP if cum_ret > 0 else DOWN`, meaning exactly zero cumulative return was classified as DOWN when Hurst indicated trending. Fixed to return FLAT when cum_ret is zero.

- **Momentum fade GARCH z-score / confluence scoring mismatch**: Entry gate correctly used GARCH conditional z-score (threshold 2.0 sigma), but the confluence scorer always scored using raw VWAP distance (threshold 0.008%). When GARCH was active with low vol, statistically significant setups (z=2.5) with small raw deviation (0.5%) scored zero on the heaviest-weighted factor (0.25 weight), suppressing valid signals. Fixed to use GARCH z-score in confluence scoring when GARCH is active.

- **Sortino ratio denominator uses wrong count**: Both daily and trade-level Sortino ratio calculations divided the sum of squared negative returns by the count of negative returns only, instead of the total observation count. This inflated downside deviation by ~58% (at 40% negative rate), making strategies appear less risky than they actually are. Fixed to divide by total count per the standard Sortino formula.

- **Walk-forward optimization train/test boundary overlap**: In both rolling and anchored WFO modes, the train and test windows shared the same boundary date. With inclusive date filtering, this leaked 1 day of test data into the training window, biasing parameter selection. Fixed by offsetting boundaries by 1 day.

- **Fill model dead code in backtest executor**: The `fill_model` parameter (supporting volume-aware slippage) was accepted and stored but never called — the executor always fell back to simple BPS-based `_apply_slippage()`. This meant volume-aware fill simulation had zero effect on backtest results despite being configured. Integrated `fill_model.compute_fill()` into the order processing loop.

- **WebSocket stream dispatch silently drops all events**: `_dispatch_event()` compared the raw gateway feed name (e.g. `"stock_bars"`) against short names (`"bars"`), so the condition never matched and all WebSocket bar/quote/trade callbacks were silently never called. `REVERSE_FEED_MAP` existed to solve this but was never used. Applied the reverse mapping before dispatch.

- **Bar parser drops zero-volume bars**: `_parse_bars()` used Python `or` chaining to pick between dict key names, but `or` treats `0` as falsy. A legitimate zero-volume bar (common in premarket/extended hours) would fall through all alternatives, return `None`, and get silently discarded. This corrupted volume-weighted indicators (VWAP) and bar counts. Replaced with explicit `is not None` checks.

- **IV surface analyze() called with wrong arguments**: `MarketContextService.update_iv_surface()` passed `current_price`, `risk_free_rate`, and `time_to_expiry` as keyword arguments, but `IVSurfaceAnalyzer.analyze()` expects positional `term_data`, `strikes`, `call_prices` and keyword `rate`, `tte`. This would crash with `TypeError` if ever called. Fixed to extract strikes and call prices from chain data and pass correct argument names.

- **VRP realized vol not annualized**: VRP compares `(VXX/10)^2` (annualized implied variance, ~1-10) against `realized_vol^2` (per-bar EWMA, ~1e-8). The 8 orders of magnitude difference made the RV component meaningless — VRP tracked only VXX price, not the implied-vs-realized spread. Fixed by annualizing per-bar realized vol (`* sqrt(252*390)`) before squaring.

- **CVaR GPD formula wrong sign — understates tail risk ~15%**: The GPD-based Expected Shortfall formula added the correction term instead of subtracting it (return-space vs loss-space sign convention). This understated tail risk by ~15%, causing the CVaR sizer to allow ~18% larger positions than justified during elevated tail risk — precisely when it should be most protective.

- **CPPI floor decay applied per-call instead of per-day**: `floor_decay_rate` was documented as "Daily floor decay rate" but applied on every `update_equity()` call (~390 times/day for 1-min bars). A 1% daily decay became 98% actual daily decay, destroying CPPI drawdown protection within a single session when enabled.

- **RSI bounce GARCH z-score passes dollar deviation instead of fractional return**: `_compute_zscore()` passed `price - mean` (dollar units, e.g. $2.50) to `conditional_zscore()` which divides by `conditional_vol` (decimal return units, ~0.015). This produced z-scores ~167x too large, making the GARCH path trivially pass any threshold and effectively disabling it as a discriminator. Fixed to convert deviation to fractional return units (`deviation / mean`).

- **Strategies run before exit checks — conflicting orders on same bar**: `_run_strategies()` executed before `_manage_positions()`, so a strategy could submit a new entry order while the position manager was about to trigger an exit on the same symbol. Swapped the order: exits now process first.

- **Position mismatch leaves stale local positions generating phantom exits**: When a local position was missing at the broker, `_handle_position_mismatch()` set risk mode to "off" but left stale `symbol_state.position` intact. The PositionManager continued evaluating exit logic and submitting exit orders against non-existent broker positions. Fixed to clear stale local positions.

- **Fill dedup set cleared entirely at 10k — reprocessing window**: When `_processed_fill_ids` exceeded 10,000 entries, the entire set was cleared, making all previously-seen fills eligible for re-processing on WebSocket reconnection. Fixed to evict oldest half instead of clearing all.

- **MAE/MFE uses current qty instead of initial qty after partial exits**: `_update_mae_mfe()` computed `risk_per_share = open_risk / pos.qty` which inflates after partial exits (qty decreases but open_risk stays constant), distorting R-multiple metrics. Fixed to use `initial_qty`.

- **Monte Carlo Sharpe annualization uses wrong factor**: Used `min(n_trades, 252)` as the annualization factor, assuming all trades occurred in one year. For multi-year backtests this under-annualizes; for sub-year backtests this over-annualizes. Added optional `calendar_days` parameter to compute proper trades-per-year.

- **Risk sizers force minimum 1 share, bypassing risk reduction**: CPPI, CVaR, and momentum crash sizers used `max(1, int(qty * multiplier))`, preventing them from reducing position size to zero even in extreme conditions. This contradicted the regime gate which correctly rejected when sizing went below 1. Fixed to reject signals (return 0) when risk-adjusted sizing goes below 1 share.

- **Pair trading OU estimator double-updated per bar**: `_compute_spread_zscore()` and `_process_pair()` both called `ou_estimators[key].update(spread)`, feeding each spread value twice. Consecutive identical values inflate autocorrelation, underestimate theta, and overestimate half-life — causing the half-life gate to reject valid mean-reversion opportunities. Removed the update from `_compute_spread_zscore()`.

- **TrendRiderPro Hurst exponent double-updated**: `_update_quant_state()` called `hurst.update(close)` per bar, then `generate_signal()` called it again for metadata. Double-feeding prices corrupts R/S analysis. Fixed to recompute from cached prices without appending.

- **Hurst exponent computed on raw price diffs instead of log-returns**: R/S analysis used `np.diff(prices)` which is not scale-invariant — a $500 stock produces 10x larger diffs than a $50 stock. Fixed to use `np.diff(np.log(prices))` (log-returns). Also fixed off-by-one using `len(prices)` instead of `len(returns)` in the R/S formula denominator.

- **Prior day stats returns wrong day before market open**: `get_prior_day_stats()` unconditionally took `bars[-2]` assuming `bars[-1]` was today's incomplete bar. Before market open, today's bar doesn't exist yet, so `bars[-1]` is yesterday's complete bar and `bars[-2]` is two days ago. This produced incorrect gap percentages for gap_fill and ORB strategies during premarket. Now checks whether the last bar's date matches today.

- **Kalman hedge ratio returns posterior residual instead of innovation**: `KalmanHedgeRatio.update()` computed the residual after the Kalman update (posterior state), not before (prior prediction). Innovation should measure how surprising the new observation is given the prior — using the posterior dampens the signal since the filter has already partially absorbed it. Fixed to compute innovation before calling `kf.update()`.

- **Johansen cointegrating vector count overcounts**: Simple summation counted all eigenvalues exceeding critical values, even after a non-rejection. The Johansen sequential testing procedure requires stopping at the first non-rejection — otherwise you overcount cointegrating relationships. Fixed to break on first non-rejection.

- **CUSUM z-score normalization includes current observation**: The CUSUM detector appended the current value to the rolling window before computing mean and std, making the z-score self-referential. This dampens detection sensitivity because the observation being tested biases the statistics toward itself. Fixed to compute mean/std from prior observations only.

- **flow_alpha dof_score=0 maps to max bearish signal**: When DOF data is missing (default 0.0), the `dof_score * 2.0 - 1.0` transform produced -1.0 (strongest bearish signal), systematically biasing flow_direction toward SELL for any symbol without directional options flow data. Fixed to treat 0.0 as neutral.

- **flow_alpha IC tracker uses same-bar correlation instead of lagged**: The Information Coefficient tracker paired current bar's signal values with current bar's return (contemporaneous correlation), not the prior bar's signals with current return (predictive correlation). This inflated IC estimates and produced unreliable adaptive signal weights. Fixed to lag signals by one bar.

- **flow_alpha target distance ignores regime volatility adjustment**: Stop distance was regime-adjusted (widened in HIGH/SHOCK vol) but target distance used un-adjusted raw risk, causing the effective R:R ratio to silently drift from the configured value. In SHOCK regime (1.5x multiplier), a configured 3R target became effective 2R. Fixed to compute target from regime-adjusted stop distance.

- **VWAP reversion fallback VWAP computed from wrong bar set**: When no pre-computed VWAP was available, the fallback calculated VWAP from all bars (potentially multi-day) while standard deviation used session-only bars. This mismatch produced bands that were too tight relative to the VWAP level, generating spurious entries. Fixed to use session bars for both.

- **Heber read client hardcodes `equity:` instrument key**: `get_bars()` and `get_trades()` always used `instrument_key=f"equity:{sym}"` regardless of the `instrument_type` set at initialization. Crypto and options reads would match no rows, returning empty results. Fixed to use `self.instrument_type`.

- **Config env var override rejects negative numbers**: `APP_*` environment variable overrides used `str.isdigit()` which returns `False` for negative numbers. Values like `"-0.5"` were stored as strings instead of floats. Fixed to strip leading `-` before digit check.

- **Drawdown throttle is always a no-op**: When no historical max drawdown data is provided, the allocator fell back to `historical_dd = current_dd`, making the check `current_dd > 1.5 * current_dd` — mathematically impossible. The drawdown safety mechanism was silently disabled for any strategy without pre-computed historical max drawdown. Fixed to skip throttling when no historical baseline exists.

- **Marginal CVaR percentage limit never enforced**: `max_marginal_cvar_pct` was accepted, stored, and `marginal_cvar` was computed, but no conditional check was ever performed. A single position could consume an arbitrarily large fraction of the total CVaR budget. Added the missing check before the absolute budget gate.

- **Pair trading leg 2 hedge ratio hardcoded to 1.0**: The scanner always set the second leg's hedge ratio to 1.0 regardless of the actual cointegration-derived ratio. Downstream pair execution using this metadata for position sizing would compute incorrect notional ratios, leaving the pair unhedged. Fixed to use the reciprocal (`1.0 / hedge_ratio`).

- **IV surface RND uses wrong finite difference for non-uniform strikes**: The second derivative computation used `(C[i+1] - 2C[i] + C[i-1]) / dk_avg²`, which is only correct for uniform strike spacing. Options chains have non-uniform spacing (wider away from ATM), producing incorrect risk-neutral density values. Fixed to use the proper non-uniform central difference formula.

- **EWMA volatility is not actually EWMA**: `_compute_ewma_vol()` had no persistent state — it recomputed from scratch each call using `np.mean(sq_returns)` as a "previous" value, with an alpha that shrank as the buffer grew. This produced non-stationary behavior (completely different sensitivity during warmup vs steady state). Fixed to use proper recursive EWMA with fixed alpha and persistent state variables.

- **Unsorted strikes passed to risk-neutral density computation**: `update_iv_surface()` collected strikes and call prices in whatever order they appeared in chain data, but the finite difference formula requires monotonically increasing strikes. Unsorted data produces garbage density values. Fixed to sort by strike before passing to `compute_rnd`.

- **Momentum crash spread z-score uses population std**: `np.std(spread_array)` used `ddof=0` (population std), systematically inflating z-scores and overestimating crash probability. Fixed to use `ddof=1` (sample std), consistent with the entropy analyzer.

- **EMA slope not normalized — incomparable across price levels**: `ema20_slope` was computed as raw price difference (`ema_val - ema_prev`), making it price-level dependent. A $500 stock produces 100x larger slopes than a $5 stock. Fixed to normalize as percentage change (`(ema_val - ema_prev) / ema_prev`), consistent with `distance_from_ema20`.

- **Atlas factor scores use wall-clock date in backtests**: `_append_atlas_factors()` used `date.today()` instead of the `as_of` parameter, causing future factor scores to leak into historical backtest evaluations. Fixed to pass `as_of` from the caller.

- **SnapshotManager passes plain dataclasses to SQLAlchemy**: `persist_external_snapshot` and `persist_feature_snapshot` created `ExternalSnapshotRecord` and `FeatureSnapshotRecord` (plain dataclasses) and passed them to `session.add()`, but SQLAlchemy requires ORM-mapped models (`ExternalSnapshot`, `FeatureSnapshot`). Field names also mismatched (`data` vs `data_json`). Every snapshot write silently crashed. Fixed to construct proper ORM model instances.

- **Signal aggregator correlation penalty non-deterministic**: Strategy iteration order depended on set ordering, which is non-deterministic in Python. The same inputs could produce different penalty results across runs. Fixed by sorting strategies before iterating.

- **WFO efficiency ratio hides failed windows**: Windows where the strategy produced no trades or negative expectancy (scores <= -100) were silently dropped from the OOS average, inflating the efficiency ratio. Fixed to include all windows with failed scores replaced by 0.

- **WFO param stability uses population variance**: With typical 3-6 WFO windows, population variance (N) underestimates true variance by up to 18%, making unstable parameters appear stable. Fixed to use sample variance (N-1).

- **ORB v2 CUSUM/VR state leaks across sessions**: The CUSUM detector and Variance Ratio calculator accumulated values from prior trading days but were never reset at session boundaries. Yesterday's CUSUM cumulative sum (relative to yesterday's range midpoint) could cause false breakout detection or suppress legitimate breakouts on today's fresh range. Fixed to reset both on daily session reset.

- **Index mean reversion reports hardcoded z-score in signal metadata**: Signal meta always reported `z_score: -2.0` (long) or `z_score: 2.0` (short) regardless of actual price deviation. Fixed to compute and report the actual z-score.

- **Antifragile regime overrides discard liquidity/complexity multipliers**: `_apply_antifragile_overrides` computed the class-specific vol/risk multiplier from scratch but discarded the `base_combined` parameter entirely, losing liquidity and complexity axis adjustments. Convex strategies in STRESSED liquidity would trade at full size instead of reduced. Fixed to preserve non-overridden axes while replacing vol/risk with class-specific values.

- **Hurst exponent uses biased single-scale estimator**: `_compute_hurst()` in `regime.py` used single-scale R/S analysis (`H = log(R/S) / log(n)`), which is known to be upward-biased for short series. This over-classified FLAT markets as trending (UP/DOWN), sending incorrect trend regime signals to strategy activation policies. Replaced with multi-scale R/S regression matching the implementation in `src/quant/statistics.py`.

- **Yang-Zhang volatility includes fabricated overnight return**: The first bar's overnight return was hardcoded to `0.0` (no prior close), biasing the overnight variance estimate downward. For small lookback windows (20 bars), this pulls the Yang-Zhang estimate down by ~5%. Fixed to exclude the first bar from overnight return statistics.

- **Lempel-Ziv complexity normalization uses wrong log base**: The LZC upper bound formula used `ln(n)` instead of `log2(n)`, underestimating normalized complexity by ~31%. Fixed to use `log2(n)` per Lempel & Ziv (1976).

- **GARCH fallback volatility uses population std**: The rolling-std fallback path used `np.std()` with default `ddof=0` (population), slightly underestimating volatility. Fixed to use `ddof=1` (sample) for consistency.

- **ClosedTradeInfo PnL misses partial exit profits**: When a position had multiple partial exits, `pnl_gross` only reported the last fill's PnL, not the accumulated total. A trade with $100 from partials + $50 from final exit would report $50 instead of $150. Also affected `pnl_net`, `pnl_r`, and downstream risk manager daily PnL tracking. Fixed to use total `realized_pnl`.

- **ClosedTradeInfo qty reports last fill size instead of total**: After partial exits, the recorded trade quantity was the remaining fill (e.g., 50 instead of 100), misrepresenting trade size in analytics. Fixed to use `initial_qty`.

- **Partial exit fraction uses current qty instead of initial**: With levels `[(1R, 0.5), (2R, 0.5)]`, each level took 50% of *remaining* qty (geometric: 50, 25) instead of 50% of *initial* qty (linear: 50, 50), leaving 25 stranded shares with no exit level. Fixed to use `initial_qty` as the fraction base.

- **Sweep count inflated by bullish sentiment trades**: `_process_single_flow_trade` counted any BULLISH-sentiment trade as a sweep (`"sweep" in tags or sentiment == "BULLISH"`). Non-sweep bullish trades inflated `sweep_count` and artificially boosted DOF scores via the sweep multiplier. Fixed to count only actual sweep-tagged trades.

- **GEX flip distance picks first zero-crossing by strike order, not nearest to spot**: With multiple gamma exposure zero-crossings, the code selected the first in ascending strike order rather than the one nearest to the current price. For gamma-pinning analysis, the nearest crossing to spot is the relevant one. Fixed to select the crossing closest to spot price.

- **Bar field and volume extraction treat zero as missing**: `_get_bar_field` and `_extract_volume` in fetcher.py used Python `or` for fallback, which treats `0`/`0.0` as falsy. Legitimate zero-volume bars (premarket) would get wrong values from fallback keys. Fixed to use explicit `is not None` checks.

- **Stream bar vwap/trade_count treat zero as missing**: `_normalize_stream_bar` in client.py used `or` for vwap and trade_count fields, dropping valid zero values. Fixed to use `is not None` pattern matching the OHLCV fields.

- **Backtest VWAP aggregation uses simple mean instead of volume-weighted**: When pre-computing 5m/15m indicators, VWAP was aggregated as `mean` of 1m VWAPs, treating all bars equally regardless of volume. A 1M-share bar and a 100-share bar contributed equally. Fixed to use `sum(vwap * volume) / sum(volume)`.

- **Backtest higher-TF indicator look-ahead bias**: Before the first higher-TF bar completed (e.g., first 4 bars for 5m), the index was clipped to 0, reading from a bar whose OHLCV included future 1m data. Fixed to return no indicators until the first higher-TF bar has fully completed.

- **Ruff lint errors**: Fixed 6 extraneous f-prefixes in `scripts/run_wfo_robust.py`.

- **TrendRiderPro `_regime_allows` missing method**: Added the `_regime_allows` static method to `TrendRiderProStrategy`. BUY signals are now only allowed when the trend regime is UP, SELL signals only when DOWN, and all signals pass when no regime snapshot is available (backward compatibility). Fixes 3 failing unit tests.

- **Dev dependencies not installed in venv**: The Cerberus virtualenv was missing pytest and other dev extras, causing `uv run pytest` to fall back to the system conda pytest which lacked `filterpy`. Running `uv sync --extra dev` resolves this; all 1271 tests now pass.

### Added

#### Backtest & WFO Robustness

- **FastAPI backtest API**: New REST API at `src/api/backtest_api.py` exposes backtest runs, equity curves, trades, Monte Carlo results, regime splits, and WFO parameter sensitivity for dashboard consumption. CORS configured for EmpireUI (localhost:5173).

- **Diagnostics engine wired into backtest pipeline**: When `analytics.diagnostics.enabled: true` is set in config, the backtest runner now runs post-backtest diagnostics (strategy rankings, regime mismatches, time-of-day edge map, hold/exit analysis) and attaches the results to the report card. Summary is logged and included in JSON output.

- **JSON result persistence**: Backtest results are now saved as JSON files, enabling the dashboard API to list and retrieve past runs. Use `save_backtest_result()`, `load_backtest_result()`, and `list_backtest_runs()` from `src/backtest/result_store`.

- **Holdout validation dataclass**: Added `HoldoutResult` dataclass to the WFO harness for structured holdout window validation results, including OOS-to-holdout ratio and pass/fail status.

- **Parameter sensitivity in WFO results**: Walk-forward optimization now runs parameter sensitivity analysis on completed Optuna trials and includes ranked sensitivity data in the returned results under `param_sensitivity`.

- **Fill model config wiring**: Added `create_fill_model()` factory function that builds the appropriate fill model (fixed or volume-aware) from backtest config. The backtest runner now passes the constructed fill model to `SimulatedOrderExecutor`.

- **Data quality checks in backtest**: The backtest runner now runs data quality checks after loading bars, excluding symbols with insufficient coverage and logging warnings for gaps, outliers, and stale prices. Configurable via `analytics.data_quality` in config.

- **Benchmark comparison in backtest report**: The backtest runner now computes benchmark comparison (alpha, beta, information ratio, capture ratios) against SPY after each backtest and includes it in the markdown report and JSON output.

- **Monte Carlo simulation in backtest report**: New `src/analytics/monte_carlo.py` module runs bootstrap resampling of trade P&Ls. Enable via `analytics.monte_carlo.enabled: true` in config to get probability of loss/ruin, equity confidence intervals, and Sharpe distribution in the backtest report.

- **Post-backtest diagnostics engine**: New `src/analytics/diagnostics.py` module runs five analyses after a backtest — strategy ranking by P&L, regime mismatch detection, time-of-day edge mapping, hold/exit-type analysis, and a plain-text summary. Use `run_diagnostics(trades)` with a list of trade dicts.

- **Benchmark comparison analytics**: New `src/analytics/benchmark.py` module computes alpha, beta, information ratio, and up/down capture ratios against a benchmark (e.g., SPY). Use `compute_benchmark_comparison()` with daily return arrays.

- **Per-strategy overnight position handling**: Added `allow_overnight`, `max_hold_days`, and `overnight_stop_mult` fields to `BaseStrategy._set_params()` for configuring overnight hold behavior per strategy instead of globally.

- **Strategy name on Position**: Added `strategy_name` field to `Position` dataclass, set automatically when a new position is opened via `PositionManager`.

- **Per-strategy EOD flatten logic**: The backtest runner's `force_flat_at_1600` now checks each position's strategy overnight config individually instead of flattening all positions globally. Strategies with `allow_overnight=True` will hold positions overnight unless `max_hold_days` is exceeded.

- **FillModel protocol and FillResult dataclass**: Added pluggable fill model interface (`src/backtest/fill_models/protocol.py`) with a `FillModel` runtime-checkable Protocol and a frozen `FillResult` dataclass. This is the foundation for swappable fill simulation in the backtest engine.
- **FixedSlippageFillModel**: Extracted the fixed-BPS slippage logic from `SimulatedOrderExecutor` into a standalone `FixedSlippageFillModel` class that satisfies the `FillModel` protocol. Supports configurable slippage (basis points) and per-share commission.
- **VolumeAwareFillModel**: Added volume-aware fill model that scales slippage based on order participation rate (order size / bar volume). Supports configurable base slippage, impact coefficient, and max slippage cap. Replaces fixed-BPS slippage as the more realistic default for backtesting.
- **Pre-backtest data quality checker**: Added `DataQualityReport` and `check_data_quality()` to validate bar data before replay. Detects gaps, zero-volume bars, price outliers, stale prices, and low-coverage symbols with configurable thresholds and automatic exclusion.

- **Per-window HMM retraining in WFO**: The walk-forward optimizer now retrains the HMM regime model on each training window's daily bars before running the IS optimization and OOS backtest. This eliminates lookahead bias from using a globally-trained HMM during WFO. Artifacts are saved per-window under the run's artifact directory.

- **Backtest Parity Improvements**: Enhanced backtest realism with configurable simulation settings
  - Volume-aware partial fills: `partial_fill_mode` (none|fixed|volume_aware) with `partial_fill_rate` for liquidity modeling
  - Volume-impact slippage: `slippage_mode` (fixed|volume_impact) with `slippage_impact_mult` for market impact simulation
  - ATR-based spread: `spread_mode` (fixed|atr_based) for volatility-sensitive spread modeling
  - Flow strategy gating: `disable_flow_strategies` config to skip flow-dependent strategies in backtest
  - New `backtest:` config section in `config.yaml` with all realism settings
  - Unit tests in `tests/unit/test_backtest_parity_unit.py` (13 new tests)

- **Backtest Session VWAP Injection** (`src/backtest/runner.py`):
  - Added `_vwap_state` tracking dict for cumulative TPV/volume per symbol per session
  - Session VWAP calculated and injected as `bar.vwap` attribute for VWAP-based strategies
  - Enables `vwap_trend_rider` and `vwap_reversion` in offline backtests

- **Backtest Gap Calculation** (`src/backtest/runner.py`):
  - Added `_prev_day_closes` tracking for gap percentage calculation
  - `gap_pct` injected into `symbol_state.meta` for `gap_fill` strategy

- **Enhanced Regime Analysis Script** (`scripts/analyze_regime_ci.py`):
  - Wilson score 95% confidence intervals for statistically robust win rate estimation
  - Confidence ratings based on sample size (insufficient/low/medium/high)
  - High-confidence deployment recommendations (CI lower > 25%)

#### Strategy Upgrades

- **Optuna-tunable quant gate thresholds**: Made hardcoded quant gate thresholds (Hurst, HMM confidence, BOCPD, kurtosis, entropy) into Optuna-optimizable parameters across all 5 active strategies. This allows WFO to find the right gate sensitivity per strategy instead of using fixed thresholds that were too restrictive (mean_reversion_pro had 0 trades in 11/21 WFO windows). The flat `hmm_min_confidence` param is automatically wired into the nested `hmm_gate.min_confidence` config structure by the optimization harness.

- **HMM regime gate**: Added `_check_hmm_gate()` to BaseStrategy that reads HMM predictions from `market_state.meta["hmm_regime"]` and rejects trades when the predicted regime matches strategy-specific rejection rules. Wired into all 5 active strategies (mean_reversion_pro, trend_rider_pro, orb_v2, rsi_bounce, momentum_fade) with per-strategy reject lists and confidence thresholds configured in `backtest_v2.yaml`.

- **Strategy data_requirements declarations**: Added `data_requirements` class attribute to `BaseStrategy` (default: bars stream only) and overrides on all 12 strategies that need non-default feeds. Strategies now declare their WebSocket streams (bars/quotes/trades) and on-scan REST fetches (flow/gex/prior_day) so the engine can auto-subscribe to the right data.

#### Quant Foundation

- **Quant foundation layer** (`src/quant/`): GARCH conditional volatility, Kalman filters, Engle-Granger/Johansen cointegration, Hurst exponent, CUSUM breakout detection, Granger causality, Markov regime-switching, adaptive thresholds, walk-forward validation, deflated Sharpe ratio
- **Portfolio optimization layer** (`src/portfolio/`): IC-weighted signal aggregation, risk-parity allocation with drawdown throttle, portfolio VaR/CVaR with concentration limits, strategy attribution with rolling Sharpe/Sortino
- New dependencies: `arch>=7.0`, `filterpy>=1.4`
- New DB tables: `strategy_ic_daily` (daily IC tracking per strategy), `portfolio_risk_snapshots` (point-in-time VaR/CVaR/concentration)
- **momentum_fade quant upgrade (1.5/5 to 3.5/5)**: Replaced raw VWAP deviation threshold with GARCH-conditional z-score (falls back to raw % when GARCH not fitted), added momentum exhaustion model (velocity + deceleration gate — only fades when momentum is decelerating), Hurst exponent gate (rejects entries when H >= 0.5, indicating trending regime), entropy filter (rejects when regime entropy > 0.8), and intraday seasonal volume adjustment (15-minute time-of-day buckets replace flat 20-bar average). Confluence factor weights rebalanced to include GARCH z-score intensity and exhaustion score factors.
- **rsi_bounce quant upgrade (4/5 to 4.5/5)**: Added GARCH-conditional z-score (replaces rolling std when GARCH model fitted), BOCPD structural break gate (rejects entries when changepoint_probability > 0.7), rolling kurtosis filter (rejects entries when excess kurtosis > 6, indicating fat-tailed instability), and AdaptiveThresholdEngine for z_entry and confluence thresholds (scales by GARCH vol and OU half-life). All changes are surgical additions — existing 6-factor model and gates preserved.
- **flow_alpha quant upgrade (2.5/5 to 4/5)**: Replaced static signal weights (0.35/0.25/0.20/0.20) with IC-weighted combination that adapts to predictive power of each flow signal. Added Granger causality validation (tests flow_zscore → returns every 500 bars, reduces conviction 30% when not causal). Added VPIN toxicity gate (rejects entries during informed-flow periods). Replaced fixed z-score /3.0 normalization with GARCH-conditional volatility scaling. All quant components are lazy-initialized per symbol.
- **orb_v2 quant upgrade (1.5/5 to 3.5/5)**: Replaced naive price-threshold breakout with CUSUM change-point detector (statistical breakout confirmation), added Lo-MacKinlay variance ratio gate (suppresses breakouts when VR indicates mean-reversion), integrated BOCPD changepoint probability from regime system as confluence factor (weight 0.15), and replaced fixed volume gate with GARCH-conditional volume scaling. All quant components are lazy-initialized per symbol and fed during range-building and evaluation phases.
- **mean_reversion_pro quant upgrade (2.5/5 to 4/5)**: Added GARCH(1,1) conditional z-score for VWAP distance (replaces raw distance when model is fitted), Hurst exponent gate (rejects entries when H >= 0.45, indicating non-mean-reverting regime), OU half-life hard gate (rejects when half_life > max_hold_minutes), and AdaptiveThresholdEngine for confluence scoring (scales threshold by volatility and Hurst). All quant components are lazy-initialized per symbol and updated every bar.
- **trend_rider_pro quant upgrade (2/5 to 3.5/5)**: Replaced EMA-20 pullback detection with Kalman mean tracker (adapts to trend speed, no fixed lag), added Hurst exponent trending gate (rejects entries when H <= 0.55), replaced hardcoded ADX threshold with Markov regime probability (requires filtered_probability > 0.7), and replaced fixed ATR stops with GARCH conditional volatility stops. All quant components are lazy-initialized per symbol and updated every bar.
- **pair_trading_v2 quant upgrade to 5/5**: Replaced EMA hedge ratio with Kalman filter (KalmanHedgeRatio), added Engle-Granger cointegration gate (rejects entries when p_value > 0.05), GARCH-conditional z-scores (replaces rolling std when GARCH model available), OU half-life hard gate (rejects entries when half_life > max_hold_minutes), and rolling correlation monitoring per pair. All gates applied before confluence scoring.
- **Lo-MacKinlay variance ratio test module** (`src/analysis/variance_ratio.py`): Implements VR(k) with heteroscedasticity-robust z-statistic for detecting mean-reverting vs trending price regimes. Zero external dependencies (uses pure-Python normal distribution helpers). Includes 29 unit tests covering statistical properties, edge cases, and numeric helpers.
- **RSI Bounce v2 — institutional-grade mean reversion**: Upgraded from 3-factor (RSI + BB + trend context) to 6-factor confluence model: z-score extremity, OU half-life validity (primary gate), RSI percentile rank, volume climax, momentum deceleration, and Lo-MacKinlay variance ratio gate. Adds VPIN toxicity filter to skip entries during informed trading. Drop-in replacement with same class name and strategy interface.

#### Data Client & Streaming

- **UnifiedDataClient WebSocket streaming**: Added WebSocket methods to `UnifiedDataClient` for real-time bar, quote, and trade streaming via Data-Gateway. Includes `connect()`/`disconnect()`, `subscribe()`/`unsubscribe()`, `update_subscriptions()` with delta diffing, `start_stream()` with auto-reconnect and exponential backoff, and `StreamQuote`/`StreamTrade` dataclasses. Feed name mapping (`bars` -> `stock_bars`, etc.), heartbeat handling, and both sync/async callback dispatch. 25 tests in `tests/data/test_unified_client_ws.py`.

- **UnifiedDataClient (REST)**: Added `src/data/client.py` with `UnifiedDataClient` class providing all REST methods for Data-Gateway communication. Replaces `CentralApiClient` with a cleaner interface: retry logic (429/5xx with exponential backoff, immediate raise on 401/403), bar/trade normalization, order management, screener endpoints, and computed helpers (`get_prior_day_stats`, `get_avg_daily_volume`). WebSocket URL computed and stored for future streaming support. 37 unit tests in `tests/data/test_unified_client_rest.py`.

- **DataRequirements dataclass**: Added `src/data/requirements.py` with `DataRequirements` dataclass and `aggregate_requirements()` function for strategies to declare needed data feeds (bars, quotes, trades streams and on-scan REST fetches like flow/gex). Part of the unified data client migration.

- **Dynamic Ticker Discovery**: True live-parity stock discovery using Alpaca Screener API
  - `AlpacaClient.get_most_actives()` - Fetch top volume stocks
  - `AlpacaClient.get_movers()` - Fetch top gainers/losers
  - `UniverseBuilder` screener dynamic source with configurable `most_actives_top_n` and `movers_top_n`
  - `scripts/capture_screener_snapshot.py` - Daily snapshot capture for future historical replay
  - Setup guide: `docs/screener_snapshot_setup.md`

#### Ledger & Persistence

- **Ledger Adapter**: Added `CerberusLedgerAdapter` (`src/core/ledger_adapter.py`) to record all trade opens and closes in the unified empire-core ledger. Maps Cerberus-specific fields (regime tags, R-multiples, MAE/MFE, features) to the standardized ledger schema. Back-fills open records for trades that started before the adapter was initialized. Integrated into `PositionManager` with best-effort try/except so ledger failures never interrupt trading.

- **Ledger adapter wired into execution flow**: `ExecutionEngine` now creates a `CerberusLedgerAdapter` and passes it to `PositionManager`, so all trade opens and closes are recorded in the unified ledger (`ledger.db`). Adapter creation is wrapped in try/except so failures never break trading. The ledger DB path is configurable via `ledger_db_path` in the engine config (defaults to `ledger.db`).

#### Integration & Tooling

- **Cerberus/Data-Gateway/Heber integration gate tooling** (2026-02-11):
  - Added one-command integration smoke script:
    - `scripts/smoke_gateway_heber_integration.py`
  - Added unit coverage for smoke gate checks:
    - `tests/unit/test_smoke_gateway_heber_integration_unit.py`

- Added local Claude/Swarm workspace tooling assets and skill bundles:
  - `.claude/` helpers, settings, and skill definitions
  - `.claude-flow/` agent/task state files
  - `.swarm/` runtime state files
  - `CLAUDE.md` and `vectors.db` local support artifacts

- **Data-Gateway/Heber Phase 1 Completion** (2026-02-10):
  - Enhanced dual-read parity logging with comprehensive comparison:
    - Bar value comparison (OHLCV) with percentage difference tracking
    - Trades count parity logging for gateway vs legacy
    - Flow count parity logging for gateway vs legacy
    - GEX data parity confirmation logging
    - Success logs for confirmed parity across all data types
  - Startup environment validation for gateway/heber modes:
    - Added `validate_startup_mode()` method to `Settings` class
    - Added `validate_startup_settings()` function for main entry point validation
    - Validates required env vars based on configured backend mode:
      - Gateway mode: requires `CERBERUS_GATEWAY_URL` and `CERBERUS_GATEWAY_KEY`
      - Heber mode: requires `CERBERUS_HEBER_CATALOG_URL`
      - Legacy/dual+failover: requires Alpaca credentials
    - Integrated validation into `src/main.py` startup sequence
  - Comprehensive gateway/failover integration tests:
    - Created `tests/integration/test_gateway_failover_integration.py` with 11 test scenarios
    - Created `tests/unit/test_startup_validation_unit.py` with 14 validation tests

- Automated hourly report generated (2026-02-13 04:03 UTC).

- **Config: pydantic-settings for runtime env vars** (2026-02-09)
  - Created `src/core/settings.py` with `Settings(BaseSettings)` for Alpaca credentials
  - Migrated `health.py` from `os.getenv` to settings with `resolved_*` property helpers
  - Supports both `ALPACA_*` and `APCA_*` naming conventions via resolved properties
  - Added `pydantic-settings>=2.0` dependency

#### Alpha Overhaul

- **Alpha Overhaul Phase 4: Order Flow & Microstructure**:
  - `Trade Flow Imbalance (TFI)`: High-fidelity microstructure edge using Tick Test (Lee-Ready).
  - `Net Gamma Exposure (GEX)`: Integrated Unusual Whales greek exposure API for MM pinning analysis.
- **Alpha Overhaul Phase 5: Statistical & Regime Alpha**:
  - `Fractional Differentiation`: achieved stationarity while preserving memory ($d \approx 0.4$).
  - `Hurst Exponent`: R/S analysis for regime classification (MR vs Trending).
- **Alpha Overhaul Phase 6: Meta-Labeling & Probabilistic Execution**:
  - `Signal Enrichment`: Every `Signal` now carries a `feature_snapshot` representing the full alpha context at time of generation.
  - `MetaLabeler`: Implementation of a heuristic v1 vetteur using Hurst, TFI, and GEX to reject low-probability trades.
  - `Database Persistence`: Enhanced `signals` table schema to log feature snapshots for future model training.
- **Alpha Overhaul Phase 7: Automated Parameter Tuning & Walk-Forward**:
  - `Dynamic Parameter Updates`: Refactored `BaseStrategy` and all strategies to support `update_params` for runtime parameter injection.
  - `GridSearchOptimizer`: Modular parameter search with custom scoring functions.
  - `WalkForwardManager`: Rolling window stability checks to prevent overfitting.
  - `Stage2Tuner Integration`: Enhanced agent tuning with walk-forward validation.

#### Historical Replay & Snapshot

- **Historical Replay Data Architecture**:
  - `ExternalSnapshot`: Layer 1 table for raw API data (GEX, flow) capture.
  - `FeatureSnapshot`: Layer 2 table for computed features at point-in-time.
  - `DailyUniverse`: Tracks which symbols passed filtering each day.
  - `SnapshotManager`: Orchestrates capture of external API data and computed features.
  - `ReplayProvider`: Provides historical data from snapshots for offline backtesting.
  - `FeaturePipeline Integration`: Auto-persists snapshots when `snapshots.enabled: true`.

#### Agent & Scheduling

- **Agent Stage 2 Pipeline Integration** (PRD 9.2): Parameter tuning now runs as part of daily agent cycle in `run_cycle_with_db()`. Strategies not disabled by Stage 1 are evaluated for parameter improvements.
- **Agent Stage 3 Weekly Analysis** (PRD 9.3): New `run_weekly_analysis()` method generates weekly performance reports with LLM-powered feature/model recommendations. Reports saved to `artifacts/weekly_reports/`.
- **Scheduler Weekly Job**: Added Friday 16:30 ET job for automatic Stage 3 weekly analysis.
- **Stage 2 Search Space Config**: Added parameter search space for `vwap_reversion`, `orb`, `gap_fill`, `vwap_trend_rider`, and `index_mean_reversion` strategies.

#### Schema & Regime

- **Multi-Axis Regime Schema** (PRD Regime Upgrade Patch §7):
  - Trade table: Added `regime_tags_entry_json` and `regime_tags_exit_json` columns for full regime context
  - RegimeHistory table: Added `model_version`, `trend`, `vol_regime`, `liquidity`, `risk`, `session`, `vol_of_vol`, `liquidity_score`, `risk_score`, `confidence_json` columns
- **Signal Fusion Core (Phase 2)**: Added propagation of `atr`, `orb_high`, `orb_low`, `dof_score`, and `relative_strength` from features to strategy execution metadata.
- **Relative Strength (RS) Calculation**: Implemented benchmark-anchored RS calculation in `BacktestFeaturePipeline` and `FeaturePipeline`.
- **Directional Options Flow (DOF) Support**: Added skeleton and metadata mapping for DOF/UW flow features in `SymbolFeatures`.
- **Scanner Profiles**: Added `VixSpikeFadeProfile` and `MomentumContinuationProfile` to complete scanner coverage for all active strategies
- **Statistical Dependencies**: Added `statsmodels==0.14.4` to `requirements.txt`.

#### Error Logging & Observability

- **Error Logging Improvements**: Comprehensive audit and enhancement of error logging across the codebase
  - Added `exc_info=True` to 16 critical ERROR-level logs for full stack traces in production debugging
  - Added DEBUG-level logging to 5 silent exception handlers for best-effort operation visibility
  - Expanded ErrorCode enum from 15 to 50+ codes organized by category (Config, Analytics, Alpaca, Engine, Scanner, Risk, Orders, Agent, Database, Backtest)
  - Improved production debugging capability, observability, and error categorization for operational monitoring
  - Commits: `5eb2db6`, `b7b7788`, `61fcd7b`

#### Repository Hygiene

- **Repository Hygiene (PR #1)**: Added project identity files for open-source readiness
  - LICENSE file (MIT License) for legal clarity
  - SECURITY.md with vulnerability disclosure policy and trading-specific security guidelines
  - .env.example template with safe defaults and comprehensive documentation
  - Updated README.md to reference LICENSE, SECURITY.md, and .env.example
- **Repository Hygiene (PR #2)**: Reorganized root-level utilities for clarity
  - Created `tools/` directory with comprehensive README
  - Moved `verify_architecture.py`, `verify_deepseek.py`, `paper_live_harness.py` to tools/
  - Archived obsolete `codereview_notes.md` to artifacts/archive/
- **Repository Hygiene (PR #3)**: Added operational maturity tooling
  - Created `docs/runbook.md` with 6 failure scenarios, diagnostics, and recovery procedures
  - Implemented `src/core/health.py` with database/API/system health checks
  - Added `--healthcheck` CLI flag for operational readiness verification
  - Updated README.md with healthcheck usage documentation

#### Strategies (Original Suite)

- **Strategies**: Implemented full suite of 8 remediation strategies:
  - VWAP Mean Reversion
  - Opening Range Breakout (ORB)
  - Trend Pullback
  - Failed Breakout Fade
  - VWAP Trend Rider
  - Index Mean Reversion
  - Flow-Confirmed Momentum
  - Gap-Fill Scalper
- **Scanner**: Implemented `ScannerProfile` interface and specific profiles for all 8 strategies. Filters based on technicals (ADX, RSI, BB) and Option Flow (Unusual Whales Z-Score).
- **Pipeline**: Added comprehensive feature generation:
  - `prior_day_high`, `prior_day_low`
  - `bb_upper`, `bb_lower`, `price_zscore`
  - `flow_zscore`, `call_put_ratio` (Unusual Whales)
  - `premarket_volume` calculation
- **Architecture**:
  - `Agent` meta-loop for daily analysis and config updates.
  - `Analytics` layer for trade statistics and efficiency auditing.
  - `Scheduler` integration for automated functionality.
- **Testing**: Added unit tests for all strategies (`tests/test_strategy_*.py`).
- **Docker**: Added `Dockerfile`, `.dockerignore`, `docker-compose.yml` and `make` targets (`up`, `down`, `logs`) for full containerized orchestration.
- **Scheduler**: Added internal `APScheduler` implementation (`src/scheduler.py`) to replace external Chronos dependency. Run via `python -m src.main --scheduler`.

### Changed

#### Backtest & WFO

- **SimulatedOrderExecutor now uses FillModel protocol**: The backtest executor accepts an optional `fill_model` parameter and delegates slippage/commission calculations to it instead of using inline math. Defaults to `FixedSlippageFillModel` with the same behavior as before, making the change backward-compatible. Custom fill models can now be injected for more realistic simulation.
- **Backtest fill safeguards** (2026-02-15):
  - Defaulted to skipping zero-volume bars for fills (configurable `min_bar_volume_for_fill`).
  - Validated `partial_fill_mode` with warning fallback to `none`.
  - Added unit coverage for invalid fill modes and low-volume fill deferral.
- **Backtest Feature Parity**: Aligned `BacktestFeaturePipeline` SymbolFeatures construction with live `FeaturePipeline` to ensure consistent alpha signal availability.
- **Disabled Partial Exits** (`config/backtest_5yr/config.yaml`):
  - Set `partial_exits.enabled: false` - analysis showed early profit-taking harmed overall performance
  - Trades held >60 min had 51% WR and +$101K profit vs early exits losing money
  - Reduced slippage from 5.0 to 2.0 bps for more realistic execution modeling
- **Verification backtest results**: With only WFO-approved strategies (trend_rider_pro + orb_v2), full-year 2024 backtest achieves PF=0.91, max DD=1.96%, 1,082 trades. Strategies are near-breakeven — edge consumed by commission/friction model.
- **WFO v3 optimized params applied to trend_rider_pro and orb_v2**: Applied walk-forward optimization mean parameters with stability annotations. trend_rider_pro: confluence_threshold=63.3, stop_atr_mult=2.08, target_atr_mult=4.5, trail_min_profit_r=0.60, max_hold_minutes=138. orb_v2: confluence_threshold=54.2, vol_gate_mult=1.53, trail_min_profit_r=0.96, max_hold_minutes=93.
- **Disabled unprofitable strategies**: rsi_bounce (WFO v4 REJECT: 1/6 windows profitable, PF=0.42), momentum_fade (WFO v4 REJECT: 0 trades on 4-symbol universe), mean_reversion_pro (excluded from WFO, PF=0.72), flow_alpha (needs live options flow data).
- **Disabled all V1 legacy strategies in backtest_v2.yaml**: Added explicit `enabled: false` for 14 V1 strategies (vwap_reversion, failed_breakout, trend_pullback, etc.) to prevent ConfigLoader deep-merge from strategies.yaml activating them during V2 backtests.

#### Strategy Changes

- **Trend Rider Pro v4 — dramatic simplification for more trades**: Rewrote `trend_rider_pro` to remove ~8 gates that were suppressing trade frequency. Removed Markov regime switcher, HMM gate, Hurst gate (now observability-only), 15m HTF trend hard gate, and `_regime_allows` methods. Direction detection now uses simple VWAP filter + EMA-9/20 alignment + Kalman pullback. Confluence reduced from 6 to 4 factors, threshold lowered from 65 to 50. Default `stop_atr_mult` raised from 1.5 to 2.0, `pullback_threshold` widened from 0.003 to 0.005. Single exit config replaces HTF-strong/weak split. ADX minimum lowered from 20 to 15. Targets 2-5 trades/day across 30+ symbols.
- **Param spaces updated for Trend Rider Pro v4**: Removed `hurst_trending_threshold` and `hmm_min_confidence` params. Added `pullback_threshold` as tunable (0.002-0.008). Lowered `confluence_threshold` range to 30-60. Widened `min_trend_alignment` range to 0.15-0.55.

#### Quant Upgrade Summary

- **pair_trading_v2**: Kalman hedge ratio replaces EMA, Engle-Granger entry gate, GARCH-conditional z-score, OU half-life hard gate, rolling correlation monitor
- **mean_reversion_pro**: GARCH-conditional z-score, Hurst exponent filter (H<0.45), adaptive confluence thresholds, OU half-life hard gate
- **trend_rider_pro**: Kalman mean tracker replaces EMA-20, Hurst trending gate (H>0.55), Markov regime probability replaces ADX threshold, GARCH-forecasted stops
- **flow_alpha**: IC-weighted signal combination, Granger causality validation, VPIN toxicity gate, GARCH-conditional normalization
- **orb_v2**: CUSUM statistical breakout detection, variance ratio gate, BOCPD confidence multiplier, GARCH-relative volume gate
- **rsi_bounce**: GARCH-conditional z-score, BOCPD structural break awareness, kurtosis filter, adaptive thresholds
- **momentum_fade**: GARCH-conditional z-score, momentum exhaustion model (velocity + acceleration), Hurst gate (H<0.5), entropy filter, intraday seasonal volume

#### Data Client Migration

- **Flow alerts now read from Heber**: Option flow data is read from Heber Silver (populated by Data-Gateway's UW poller) instead of making duplicate direct API calls to Unusual Whales. Falls back to Gateway REST proxy if Heber is not configured. Set `CERBERUS_HEBER_DATA_ROOT` to enable.

- **Simplified main.py to use UnifiedDataClient exclusively**: Removed all `AlpacaClient`, `GatewayStreamClient`, and legacy data backend branching from the startup and main loop. Streaming now goes through `UnifiedDataClient.start_stream()` with automatic reconnect on session boundaries. Subscriptions update after each scan cycle via `update_subscriptions()`. Removed dead helper functions (`_should_initialize_alpaca_client`, `_should_start_alpaca_stream`, `_capture_screener_snapshot`) and their associated imports. Engine construction no longer passes `alpaca_client` or `gateway_client` params.

- **FeaturePipeline and UniverseBuilder rewired to UnifiedDataClient**: Replaced `AlpacaClient` + `CentralApiClient` params with single `UnifiedDataClient` in both `FeaturePipeline` and `UniverseBuilder`. Removed `ConfigLoader`, `get_settings()`, `use_gateway_data`, and `allow_legacy_failover` from `UniverseBuilder`. Simplified `_get_historical_bars`, `_get_screener_most_actives`, and `_get_screener_movers` to direct delegation. Updated `src/main.py`, `scripts/paper_live_test.py`, and `tools/paper_live_harness.py` construction sites. All tests updated.

- **GatewayOrderExecutor rewired to UnifiedDataClient**: Replaced `CentralApiClient` with `UnifiedDataClient` in `GatewayOrderExecutor`. Order submission now calls `unified_client.submit_order()`, cancellation calls `unified_client.cancel_order()`, and order listing calls `unified_client.get_orders()` with local symbol filtering (since `get_orders` doesn't accept a `symbols` parameter). Updated `src/main.py` to construct a `UnifiedDataClient` for the gateway executor path.

- **DataFetcher rewired to UnifiedDataClient**: Replaced `AlpacaClient`, `CentralApiClient`, and `HeberReadClient` with single `UnifiedDataClient` in `src/data/fetcher.py`. Removed all dual-read comparison methods, legacy failover logic, Heber fallback chain, and gateway/backend mode flags. File reduced from ~746 lines to ~210 lines. All data now flows through Data-Gateway via `UnifiedDataClient`. LRU cache pattern preserved for bars. 17 new tests in `tests/data/test_fetcher_unified.py`.

#### Regime & Infrastructure

- **Multi-Axis Regime Migration**: Replaced legacy BULL/BEAR/CHOP regime classification with full 5-axis multi-axis regime system
  - `Signal.regime` field removed, now uses `Signal.regime_tags: Dict[str, str]` and `Signal.regime_confidence: Dict[str, float]`
  - `Position.regime_at_entry` replaced with `Position.regime_tags_at_entry: Dict[str, str]`
  - `ClosedTradeInfo` now stores `regime_tags_at_entry/exit` dicts with 5 axes
  - Trades record full regime context: `{trend, vol, liquidity, risk, session}` at entry and exit
  - Removed legacy regime config checks from `RiskManager`
  - Updated `base.py._create_signal()` to populate regime_tags from `MarketState.regime_snapshot`
- **VXX-Based Risk Axis**: Risk axis now properly uses VXX momentum (rising VXX = RISK_OFF, falling = RISK_ON)
  - Added `update_vol(bar)` to `MarketContextService` and `MarketStateManager`
  - Wired VXX bar processing in both `BacktestRunner` and `ExecutionEngine` for parity
  - Risk distribution improved from 84% neutral to 44% neutral / 40% risk_off / 16% risk_on

#### Gateway & Execution

- **Gateway-first trading execution path** (2026-02-13):
  - Set gateway-first runtime defaults in `src/core/settings.py`:
    - `CERBERUS_DATA_BACKEND` default is now `gateway`
    - `ALPACA_PAPER` default is now `true`
  - Updated main runtime order routing in `src/main.py`:
    - `--order-executor` now defaults to `gateway`
    - Added `gateway` executor option alongside `alpaca` and `noop`
    - Blocked direct `alpaca` order execution when gateway data mode is active
  - Added gateway trading adapters in `src/data/api_client.py`:
    - `submit_alpaca_order`
    - `get_alpaca_orders`
    - `cancel_alpaca_order`
  - Added `GatewayOrderExecutor` in `src/engine/orders.py` to route submissions/cancels through Data-Gateway.
  - Updated defaults/docs:
    - `.env.example` now sets `CERBERUS_DATA_BACKEND=gateway`
    - `docs/environment-variables.md` defaults updated for `CERBERUS_DATA_BACKEND` and `ALPACA_PAPER`
  - Added coverage:
    - `tests/unit/test_gateway_order_executor_unit.py`
    - `tests/contract/test_central_api_client_contract.py` order-submit contract
    - `tests/unit/test_startup_validation_unit.py` gateway/paper defaults assertion
  - Added gateway live-stream ingestion path:
    - New `src/data/gateway_stream.py` WebSocket client for `ws://.../ws` auth + `stock_bars` subscriptions.
    - Updated `src/main.py` to stream bars via Data-Gateway when `CERBERUS_DATA_BACKEND=gateway|dual`, while retaining Alpaca stream for legacy mode.
    - Added unit coverage in `tests/unit/test_gateway_stream_client_unit.py`.

- **Central API retry classification for gateway integration** (2026-02-11):
  - Added status-aware retry policy in `src/data/api_client.py`:
    - no retry for `401/403`
    - retry for `429`, `5xx`, timeout, and transport errors
    - support for `Retry-After` with exponential backoff fallback
  - Updated checklist progress in:
    - `docs/cerberus-data-gateway-heber-implementation-checklist.md`

- Added Phase 1 integration scaffolding for Data-Gateway/Heber:
  - Extended runtime settings in `src/core/settings.py` with backend mode and Gateway/Heber config.
  - Upgraded `src/data/api_client.py` to Data-Gateway v1 routes and `X-Gateway-Key` support while preserving LLM chat compatibility.
  - Expanded `src/core/health.py` to check Data-Gateway and Heber connectivity, including gateway-mode credential handling.
  - Updated contract tests in `tests/contract/test_central_api_client_contract.py` for route and header expectations.
- Wired gateway-backed fetching in runtime data path:
  - Added Data-Gateway adapters in `src/data/api_client.py` for Alpaca trades and UW GEX.
  - Enabled `src/data/fetcher.py` to route bars/trades/flow/gex through Data-Gateway when `CERBERUS_DATA_BACKEND=gateway|dual`, with failover control via `CERBERUS_FAILOVER_TO_LEGACY`.
  - Added lightweight dual-mode parity diagnostics for bar-count mismatch in `src/data/fetcher.py`.
  - Injected `CentralApiClient` into `FeaturePipeline` from `src/main.py`.
  - Added new contract coverage for `get_alpaca_trades` and `get_uw_gex`.
- Extended gateway-backed universe sourcing:
  - Added Data-Gateway screener adapters (`most_actives`, `movers`) in `src/data/api_client.py`.
  - Updated `src/scanner/universe.py` to use Data-Gateway for dynamic volume/screener sources in gateway mode, with optional legacy failover.
  - Injected `CentralApiClient` into `UniverseBuilder` in `src/main.py`.
  - Added contract coverage for `get_alpaca_most_actives` and `get_alpaca_movers`.
- **HTTP Client: requests → httpx** — Migrated `scripts/update_universe_lists.py` from `requests` to `httpx`

#### WebSocket & Resilience

- **WebSocket Resilience Hardening** (`src/data/alpaca.py`):
  - Added explicit `feed` parameter (`DataFeed.SIP`/`IEX`) to prevent IEX fallback on premium accounts
  - Added jitter to exponential backoff (0-0.5s random)
  - Added terminal error detection (`connection limit exceeded`) to enable REST fallback
  - Limited retries to 5 with 30s max backoff
  - Stored `config_loader` for feed configuration access
  - Added resilience constants from KI: `HEARTBEAT_TIMEOUT_SEC=120`, `FIRST_BAR_TIMEOUT_SEC=10`

#### SonarQube Refactoring

- **SonarQube Refactoring**: Refactored `FlowMomentumStrategy.on_bar()` by extracting `_validate_flow_direction()`, `_get_average_volume()`, and `_build_signal()` helper methods. Reduced cognitive complexity from 26 to ~12.
- **SonarQube Refactoring**: Refactored `PositionManager.on_fill()` by extracting 8 helper methods: `_extract_fill_data()`, `_extract_risk_config()`, `_get_entry_context()`, `_apply_costs_to_position()`, `_open_new_position()`, `_increase_position()`, `_calculate_pnl()`, `_build_closed_trade_info()`, `_reduce_or_close_position()`. Reduced cognitive complexity from ~72 to ~15.
- **SonarQube Refactoring**: Refactored `PositionManager.on_bar()` by extracting 4 helper methods: `_update_mae_mfe()`, `_check_max_hold_exit()`, `_check_stop_target_exit()`, `_create_exit_intent()`. Reduced cognitive complexity from 40 to ~12.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._update_symbol_state()` by extracting 4 helpers: `_get_or_create_symbol_state()`, `_extract_current_date()`, `_handle_index_bar_update()`, `_update_session_vwap()`. Reduced complexity from 40 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._update_indicator_cache()` by extracting 5 helpers: `_collect_indicator_periods()`, `_update_ema_indicators()`, `_update_rsi_indicators()`, `_update_vol_sma_indicators()`, `_update_bb_indicators()`. Reduced complexity from 31 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._process_signal()` by extracting 5 helpers: `_bind_signal_logger()`, `_log_risk_failure()`, `_persist_signal()`, `_store_pending_entry()`, `_get_max_hold_seconds()`. Reduced complexity from 26 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._reconcile_positions()` by extracting 3 helpers: `_should_skip_reconcile()`, `_reconcile_single_position()`, `_handle_position_mismatch()`. Reduced complexity from 33 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._execute_signal_intents()` by extracting 2 helpers: `_should_halt_trading_for_db()`, `_submit_single_intent()`. Reduced complexity from 23 to ~8.
- **SonarQube Refactoring**: Refactored `ExecutionEngine.on_fill()` by extracting 3 helpers: `_normalize_fill_correlation_id()`, `_process_fill_with_position_manager()`, `_handle_closed_trade()`. Reduced complexity from 19 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine.flatten_all()` by extracting 5 helpers: `_flatten_cancel_orders()`, `_flatten_close_positions()`, `_flatten_confirm_state()`, `_flatten_reset_local_state()`, `_flatten_handle_result()`. Reduced complexity from 17 to ~8.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._refresh_strategy_engine()` by extracting 2 helpers: `_get_regime_strategies()`, `_is_strategy_enabled()`. Reduced complexity from 25 to ~8.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._process_scan_removals()` by extracting 2 helpers: `_cleanup_orders_for_symbol()`, `_get_pending_order_ids()`. Reduced complexity from 21 to ~8.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._process_scan_additions()` by extracting 3 helpers: `_build_scan_meta()`, `_determine_flow_bias()`, `_enrich_meta_from_features()`. Reduced complexity from 21 to ~6.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._reconcile_orders()` by extracting 2 helpers: `_sync_open_orders_to_state()`, `_cancel_stale_orders()`. Reduced complexity from 29 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._reconcile_db_orders()` by extracting 4 helpers: `_update_open_order_statuses()`, `_update_closed_order_statuses()`, `_mark_stale_orders_cancelled()`, `_apply_reconcile_status()`. Reduced complexity from 34 to ~8.

#### Logic Audit Improvements

- **H4 Logic Audit**: Add comprehensive fill input validation to `position_manager.on_fill()` to prevent position corruption from malformed broker data. Guards against negative quantities, zero/NaN/Inf values, and invalid types. ([#a994e0e](https://github.com/JasperDale420/Cerberus/commit/a994e0e))
- **H2 Logic Audit**: Document safe R-multiple calculation with division by zero protection for breakeven stops (initial_risk = 0). Added comprehensive test suite covering all edge cases. ([#3ec7125](https://github.com/JasperDale420/Cerberus/commit/3ec7125))
- **H3 Logic Audit**: Standardize position side comparisons to use enum (`Side.LONG`/`Side.SHORT`) consistently instead of string comparisons (`pos.side.value == "long"`). Makes code more type-safe and refactoring-friendly. ([#df32aa3](https://github.com/JasperDale420/Cerberus/commit/df32aa3))
- **H1 Logic Audit**: Add timestamp-based reconciliation race condition prevention. Position now tracks `last_updated` timestamp, and reconciliation skips positions modified within last 2 seconds to prevent fill data loss during async broker reconciliation. ([#74a1979](https://github.com/JasperDale420/Cerberus/commit/74a1979))
- **M1 Logic Audit**: Clear feature cache on market regime changes, not just daily boundaries. Prevents strategies from using stale regime-sensitive indicators (VWAP, RSI, etc.) for hours after regime transitions. Cache now clears 2-6 times per day instead of once. ([#2b88542](https://github.com/JasperDale420/Cerberus/commit/2b88542))
- **M2 Logic Audit**: Track positions carried forward at session rollover for observability. RiskManager now logs number of overnight positions at session boundaries to help diagnose position limit issues. ([#7e2a665](https://github.com/JasperDale420/Cerberus/commit/7e2a665))
- **M3 Logic Audit**: Prioritize target over stop when both exit conditions trigger on same bar. More trader-friendly since target is the better exit. Updated test to verify new behavior. ([#5514691](https://github.com/JasperDale420/Cerberus/commit/5514691))
- **M4 Logic Audit**: Skip position reconciliation for symbols with pending orders. Prevents partial fill state corruption during mid-fill broker queries. ([#dd857ee](https://github.com/JasperDale420/Cerberus/commit/dd857ee))
- **M5 Logic Audit**: Already fixed - MAE/MFE tracking happens before broker_managed_exits check, so updates on every bar
- **M6 Logic Audit**: Added optional est_exit_commission parameter to update_unrealized_pnl() for more accurate net PnL (subtracts estimated exit costs)
- **L5 Logic Audit**: Added Pydantic field_validators to RiskConfig for bounds checking: max_daily_loss (0-$100k), max_risk_per_trade (0-$10k), max_open_positions (0-100), risk_mode (normal/reduced/off). ([#a4dd8f0](https://github.com/JasperDale420/Cerberus/commit/a4dd8f0))

#### Other

- Deduplicated Stage 3 approval checks into shared helper used by weekly report and proposals.
- Feature pipeline now reuses extracted close prices per symbol to avoid duplicate passes.
  - Test coverage for: legacy mode, gateway mode, dual mode, failover behavior, parity logging
  - Tightened startup validation test precision:
    - Gateway required-field test now uses explicit empty URL value for deterministic assertions
    - Added a focused unit test confirming custom gateway URL only flags missing gateway key
- **Scanner Core**: Fixed duplicate watchlist entry bug and added sorting by score.
- **Pipeline**: Removed hardcoded `premarket_volume`; now calculates from intraday data.
- **Config**: Extended `config.yaml` to support all new strategies and parameters.
- **Agent**: Updated Stage 3 System and User prompts to be "self-annealing" and PRD-aligned, prioritizing incremental refinement over radical changes.
- **Config**: Added `unusual_whales.enabled` flags to toggle external flow data integration (disabled by default).

### Fixed

- **Replace deprecated `datetime.utcnow()` with timezone-aware `datetime.now(timezone.utc)`**: Updated 5 instances across `schema.py` (3 SQLAlchemy column defaults) and `causal_filter.py` (2 direct calls). All timestamps are now timezone-aware per Empire conventions.

- **Regime stop multiplier now applied to stop distance**: The `regime_stop_multiplier` (low=0.75, normal=1.0, high=1.5, shock=2.0) was computed and stored on positions at open time but never used to adjust the stop price. It is now applied to the stop distance when a position is opened — in high vol regimes stops are 1.5x wider, in low vol they are 0.75x tighter.

- **Test suite preflight**: `uv sync --all-extras` is now required to install dev dependencies (pytest, ruff, etc.) into the venv; previously `uv sync` alone left pytest missing, causing 4 collection errors via the system pytest missing `filterpy` in its environment.
- **MTF flat-gate tests** (`test_hierarchical_mtf_gate_unit.py`): Updated `test_rejects_when_15m_emas_diverged` and `test_rejects_when_15m_adx_high` to match the relaxed thresholds introduced in commit `9efc9d9` (EMA spread threshold 0.3% → 0.8%, ADX rejection threshold 25 → 35).
- **Risk manager notional-cap test** (`test_risk_manager_additional_unit.py`): Updated `test_risk_manager_rejects_notional_symbol_and_open_risk_caps` to reflect the capping behavior introduced in commit `5fcdce9` — when a signal's notional exceeds the per-order limit, qty is now capped to fit rather than rejected. Added a complementary assertion for the MAX_NOTIONAL rejection path (triggered when entry_price > cap, so capped qty rounds to 0).
- **Ruff lint** (34 → 0 errors): Fixed import ordering, removed unused imports and variables (`valid_pnls`, `sys`, `datetime/timezone` in scripts), removed f-strings without placeholders, renamed unused loop variables to `_sym`/`_mh`, and added `# noqa: E402` to intentional post-sys.path imports in diagnostic/runner scripts.

- **TRP `_regime_allows` rejecting FLAT trends again**: The trend_rider_pro upgrade relaxed `_regime_allows` to accept FLAT regime, but trend-following should not enter during choppy conditions. Restored strict directional gating (BUY requires UP, SELL requires DOWN). Markov upgrade path via `_regime_allows_with_markov` is unaffected.
- **`MomentumFadeStrategy` and `RsiBounceStrategy` wrong `tf_alignment_mode`**: Both strategies are mean-reversion types but inherited the base class default of `"trend"` for `tf_alignment_mode`. Added explicit `self.tf_alignment_mode = "mean_reversion"` in `_set_params` for both, consistent with all other mean-reversion strategies (gap_fill, vwap_reversion, etc.).
- **`MomentumFadeStrategy` undefined `mr_alignment` in signal meta**: The signal metadata referenced an `mr_alignment` variable that was never computed, which would cause a `NameError` at runtime when a signal was generated. Fixed by computing `mr_alignment = mtf.get_mean_reversion_alignment()` before building the meta dict.
- **Stale test for `rsi_bounce` confluence factor name**: `test_confluence_factors_in_meta` asserted `"mr_alignment"` was in the factor names, but the strategy's confluence scorer was updated to use `"trend_context"` instead. Updated the test to match the current implementation.
- **Stale test for `MeanReversionProStrategy` regime gate**: `test_rejects_trending_in_normal_vol` expected `_regime_ok` to return `False` for trending + NORMAL vol, but the regime gate was intentionally relaxed (commit 9c0a6ef) to allow NORMAL vol in trending markets. Renamed the test to `test_allows_trending_in_normal_vol` and updated the assertion to match the current behavior.

- **OCO double-fill in SimulatedOrderExecutor**: When a bar's price range spanned both take-profit and stop-loss levels, both OCO legs filled on the same bar before cancellation ran. This created negative ghost positions and double cash credits, causing equity to explode to trillions. Fixed by tracking filled parent IDs during `process_bar` and skipping sibling OCO legs.
- **Risk config key mismatch**: `SimulatedOrderExecutor` was initialized with `config.get("risk_management")` but the YAML key is `"risk"`. This caused zero slippage and zero commissions in all backtests. Now tries `"risk"` first, falls back to `"risk_management"`.
- **rsi_bounce/momentum_fade zero trades from daily trade cap**: Both strategies generated valid signals but `MAX_TRADES_PER_DAY=10` was exhausted by other strategies first. Increased to 50, added both to `strategy_routing` buckets, tuned entry thresholds (momentum_fade: confluence 45, vwap 0.5%, vol surge 1.5x; rsi_bounce: confluence 50, RSI 20/80, band tolerance 1.5%).
- **Missing strategy registrations**: `rsi_bounce` and `momentum_fade` were never registered in `_build_strategy_registry()`, so they silently produced 0 trades despite being enabled in config. Added both to the registry.
- **RSI Bounce extreme thresholds**: Default RSI oversold/overbought thresholds were 10/90, making the strategy nearly impossible to trigger. Changed to 25/75 for realistic mean-reversion signals.
- **Backtest fill flush performance**: `await asyncio.sleep(0)` after every bar was very slow (~200K yields per month). Now only yields when executor had open orders that could have filled.
- **Backtest fill dispatch chain**: `SimulatedOrderExecutor._dispatch_event` wrapped fill data in `_MockUpdate` objects that `handle_trade_update` couldn't extract, so `PositionManager.on_fill` was never called in backtests. Fixed by: (1) extracting `_MockUpdate` attributes including `fill_qty`/`fill_price` mapping, (2) adding `await asyncio.sleep(0)` after each `process_bar` call to flush async fill tasks immediately instead of deferring them. This enables max_hold exits, trailing stops, and partial exits in backtests.
- **Backtest EOD flatten timing**: Positions were held overnight because the backtest runner only flattened at the next day's first bar (9:30 AM), not at market close. Added intraday flatten at 15:55 ET when `force_flat_at_1600` is configured, matching live engine behavior. This reduced average hold times from 400-800 minutes to realistic intraday values.
- **Risk manager unit test for `MAX_SYMBOL_NOTIONAL`**: Test `test_risk_manager_rejects_notional_symbol_and_open_risk_caps` was failing because `size_hint=1` is treated as a 100% conviction multiplier (not 1 absolute share), resulting in qty=50 and notional=5000 which triggered `MAX_NOTIONAL` before reaching the `MAX_SYMBOL_NOTIONAL` check. Fixed by setting `rm.max_risk_per_trade = 1.0` before the symbol-notional assertion (yielding qty=1) and restoring it to `50.0` before the open-risk assertion.
- **Position sizing bug**: `size_hint` values between 0-1 (conviction multipliers from strategies) were truncated to 0 by `int()`, causing all signals to be rejected with `ZERO_QTY`. Now correctly treated as a scaling factor on the risk-based quantity limit, so `size_hint=0.6` means 60% of the max allowed position size.
- Sorted import block in `src/engine/execution.py` to comply with ruff I001 rule (import order within try block for ledger adapter initialization).

- **Health check: dev dependencies not installed, test suite broken** (2026-03-13):
  - Ran `uv sync --all-extras` to install dev dependencies (`pytest`, `pytest-asyncio`, `ruff`, etc.) which were missing from the virtualenv, causing all 109 test collection errors with `ModuleNotFoundError: No module named 'empire_core'`.
  - Fixed `test_structured_logger_emits_json_with_extra_fields` in `tests/unit/test_structured_logger_unit.py`: switched from `capsys` to `caplog` fixture because pytest's log capture plugin intercepts stdlib logging records before they reach the stdout handler, making `capsys` unable to see the output.
  - Added `__all__` to `src/core/logger.py` to declare the public re-export surface and fix five `F401` (unused import) ruff violations.
  - Excluded `unusualwhales_python_client-5.0.1/` (vendored third-party package) from ruff linting in `pyproject.toml` to suppress 33 spurious `UP042`/`UP046` violations in library code we do not own.
  - Result: 915 tests pass, 0 failures; ruff reports no violations.

- **Ignore temp agent DB artifacts** (2026-03-10):
  - Added `.agents/tmp/**/*.db`, `.agents/tmp/**/*.db-journal`, and `.agents/tmp/**/*.db-wal` to `.gitignore`.
  - Removed committed temporary optimization/trial SQLite databases from `.agents/tmp/` so future runs do not accumulate in git history.
- **Pre-commit** (`detect-secrets`): ignore `logs/` directories during secret scans so generated operational logs no longer block commits.

- **Deterministic WFO result discovery and per-run result labeling** (2026-03-06):
  - Updated `scripts/wfo_dashboard.py` to discover completed WFO runs from `artifacts/optimization/runs/<strategy>/<run-tag>/` instead of relying on the old flat artifact layout.
  - Added deterministic summary export to `artifacts/optimization/wfo_dashboard_summary.json` so finished WFO runs can be checked programmatically.
  - Added per-run labeled result files alongside `wfo_results.json`, for example `trend_rider_pro_<run-tag>_wfo_results.json`, so each run is stored separately even outside its run directory.
  - Added a no-`plotly` fallback so result processing still writes the HTML shell and summary JSON even when chart dependencies are missing.
  - Added regression coverage for run discovery ordering, summary payload generation, and labeled WFO result paths.

- **Backtest helper regressions** (2026-02-15):
  - Handled `None` close values when extracting mixed bar shapes in the feature pipeline.
  - Restored Alpaca stream gating based on `data_backend` in startup helpers.

- **HTTP error logging guard** (2026-02-15):
  - Avoided `ResponseNotRead` when logging streaming response bodies.

- **Critical: Zero-Trade Pipeline Fix** (2026-02-13):
  - Root cause: `_should_start_alpaca_stream()` returned `False` in `gateway+noop` mode, preventing bar WebSocket stream from starting. Without bars, `on_bar()` never fired — zero signals, zero trades.
  - Synced local `main.py` with Docker image (session control, strategy registry, market-hours helpers).
  - Gateway bar stream now always starts when `data_backend=gateway`, independent of order executor.
  - `_should_start_alpaca_stream()` scoped to only control direct Alpaca streams (executor=alpaca).
  - Added `--order-executor gateway` with `GatewayOrderExecutor` for Data-Gateway routing.
  - Changed `docker-compose.yml` default from `--order-executor noop` to `--order-executor gateway`.

- **Stability and quality gate fixes** (2026-02-10):
  - Restored compatibility imports for archived strategies via `src/strategies/failed_breakout.py` and `src/strategies/trend_pullback.py`.
  - Reinstated legacy CHOP-only guards for standalone `FailedBreakoutStrategy` and `VWAPReversionStrategy` tests.
  - Fixed `RiskManager` regime-disable rejection behavior (`REGIME_DISABLED`) and preserved rejection reason precedence when qty/risk resolves to zero.
  - Fixed signal DB persistence JSON serialization by sanitizing datetime-containing payloads before insert.
  - Normalized agent regime placeholders (e.g. `{}`) to `chop` for Stage 1 action targeting.
  - Added Stage 2 evaluator callable compatibility for deterministic test evaluators.
  - Restored flow-metric backward compatibility to 5-tuple output and kept DOF scoring in pipeline enrichment.
  - Hardened pair scanner for non-datetime price indexes and replaced statsmodels OLS dependency with numpy least-squares in pair stats/half-life calculations.
  - Made weekly scheduler job opt-in (`enable_weekly_analysis`) to keep daily-only default behavior backward compatible.

- **Critical: TechnicalFeatures Missing Field**: Added missing `last_updated` field to `TechnicalFeatures` constructor in `calculator.py`.
- **Critical: Alpaca Trade Stream Handler**: Fixed async handler compatibility with latest Alpaca SDK in `alpaca.py`.
- **Critical: Zero Signal Backtest Bug**: Fixed `BacktestFeaturePipeline` lookup window (extended to 24h) and data parity issue that prevented signal generation.
- **Critical: Scanner Return Contract**: Restored `Scanner.scan()` to return all evaluation candidates, preventing global strategy routing failures.
- **Critical: SymbolFeatures Dataclass Error**: Fixed `TypeError: non-default argument follows default argument` by reordering fields in `SymbolFeatures`.
- **Strategy Execution Fix**: Fixed missing `time` import in `FusionStrategyV1.py`.
- **Backtest Determinism**: Updated `Scanner` cache TTL to use simulated `scan_time` instead of `datetime.now()`.

- **Critical: Backtest Session Filters Now Enforced** (`src/engine/strategy_engine.py`):
  - Fixed `scanner_bypass=True` causing ALL activation policy checks to be skipped
  - Session filters (e.g., `session: [opening, midday]`) now properly block premarket trades
  - Before: 97% of trades occurred in premarket despite config filters
  - After: 0% premarket trades, proper RTH-only execution

- **H1 Logic Audit (New)**: Fix `VWAPReversionStrategy.on_bar()` unbound variable crash when price is within VWAP bands. Initialize `signal = None` before conditional blocks.
- **H2 Logic Audit (New)**: Fix `FlowMomentumStrategy` threshold logic that allowed weak flow signals. Now properly rejects all signals below `min_flow_zscore`.
- **H3 Logic Audit (New)**: Fix `BacktestAnalyzer._calculate_drawdown()` inconsistency with unrealized PnL. Peak tracking now based only on closed trades for consistency; added clarifying docstring.
- **Agent.run_cycle_with_db**: Now calls `apply_actions()` after persisting actions to DB, ensuring `strategies.auto.yaml` is automatically written.
- **Agent.apply_actions**: Fully implements REDUCE_RISK and DISABLE_STRATEGY actions: writes config to `strategies.auto.yaml`, supports both strategy-level and regime-specific overrides, adds floor at 0.0 when risk drops below threshold.
- **E2E Test Risk Values**: Update `test_prd_vertical_slice_success_metric.py` to use valid risk values within the new RiskConfig validation limits ($10k daily loss, not $1M).
- **M1 Logic Audit (New)**: Add DEBUG-level logging to silent exception handlers in `PositionManager` for MAE/MFE tracking and max-hold check failures. Improves observability without breaking trading.
- **M2 Logic Audit (New)**: Fix `RiskManager` positions_carried_forward tracking to capture count BEFORE session rollover reset for accurate logging.
- **Momentum Strategy Target Fix**: Changed target calculation in `MomentumContinuationStrategy` from bar-range-based to risk-based (`abs(entry-stop) * risk_reward`) for consistent R:R ratio.
- **M3 Logic Audit (New)**: Raise `Scanner` watchlist cap from 30 to 50 with documented PRD recommendation. Configurable limit with clearer warning message.
- **M4 Logic Audit (New)**: Add robust date extraction in `ExecutionEngine._update_symbol_state()` with clock fallback when bar_time is unusable. Prevents stale feature cache on date parsing failures.
- **M5 Logic Audit (New)**: Add `bar_duration_minutes` config parameter to `BaseStrategy` for accurate cooldown calculation across different timeframes. Defaults to 1.0 for backward compatibility.
- **L1 Logic Audit (New)**: Add named constant `QTY_EPSILON = 1e-7` in `BacktestAnalyzer` to replace magic number for floating-point quantity comparisons. Documents purpose and prevents potential infinite loops.
- **L3 Logic Audit (New)**: Add named constants `STOP_BUFFER_LONG` (0.99) and `STOP_BUFFER_SHORT` (1.01) in `FlowMomentumStrategy` for emergency stop buffer calculations. Documents the 1% buffer purpose.
- **CI Fix**: Add `asyncio_mode = "auto"` to pytest configuration in `pyproject.toml`. Enables pytest-asyncio to detect and run async test functions.
- **CI Fix**: Add missing `pytest-asyncio` dependency to `requirements.txt`. CI environment was missing this package, causing async tests to fail with "async def functions are not natively supported".
- **Memory Audit H1**: Add LRU eviction to Scanner `_feature_cache` using OrderedDict with configurable `feature_cache_maxsize` (default 1000). Prevents unbounded memory growth in long-running sessions.
- **Memory Audit H2**: Add LRU eviction to DataFetcher `_bars_cache` using OrderedDict with configurable `bars_cache_maxsize` (default 500). Evicts oldest entries when limit exceeded.
- **Memory Audit M1**: Convert ExecutionEngine `closed_trades` from unbounded list to bounded deque with maxlen=5000. Keeps last 5000 trades in multi-day runs.
- **Dead Code Removal**: Remove unused code: `ScanningError`, `run_scan_symbols()`, `run_scan_async()`, `_safe_float()`. ~150 lines removed.
- **Dead Code Removal**: Remove unused `data/models.py` (Trade/Quote classes) and test file. ~30 lines removed.
- **Indicator Consolidation**: Refactor `_compute_atr()` and `_compute_adx()` in `calculator.py` to use `RollingATR` and `RollingADX` incremental indicators. Removes ~50 lines of duplicate Wilder smoothing code.

- **Pre-commit**: Resolved all Ruff linting errors, Mypy type-check failures, and Black formatting inconsistencies across the codebase.
- **Data Pipeline**: Fix incorrect usage of `zip(strict=False)` and unused variables.
- **Testing**: Fix mock type injection errors in unit tests.

### Removed

- Removed `AlpacaClient` (`src/data/alpaca.py`), `GatewayStreamClient` (`src/data/gateway_stream.py`), and `CentralApiClient` (`src/data/api_client.py`) legacy data clients — all data now flows through `UnifiedDataClient`
- Removed legacy/dual/gateway failover integration tests, Heber shadow parity tests, and old AlpacaClient/CentralApiClient unit and contract tests
- Removed direct AlpacaClient usage from options strategies, agent evaluators, backtest runner, LLM client, and utility scripts — all rewired to `UnifiedDataClient` or simplified

- **ExecutionEngine gateway_client and alpaca_client dependencies**: Removed direct `AlpacaClient` and `GatewayStreamClient` imports, constructor parameters, and instance variables from `ExecutionEngine`. Subscription management now uses an `on_subscription_change` callback. Flatten/reconciliation uses an optional `broker_client` attribute (set externally). Order execution already routed through `order_executor` (set by `main.py`).

- **Legacy/dual/gateway backend mode settings**: Removed `cerberus_data_backend`, `cerberus_storage_backend`, `cerberus_dual_read_compare`, and `cerberus_failover_to_legacy` fields from Settings. Removed `use_gateway_data` and `use_heber_storage` properties. Simplified `validate_startup_mode()` to only check that `CERBERUS_GATEWAY_KEY` is set. All data now flows exclusively through Data-Gateway via UnifiedDataClient.

- **Scanner Profiles**: Deleted `src/scanner/profiles.py` and all ScannerProfile classes (VWAPReversion, ORB, GapFill, FlowMomentum, TrendPullback, FailedBreakout, VWAPTrendRider, IndexMeanReversion, VixSpikeFade, MomentumContinuation). Profiles were bypassed by `strategy_routing` config — all validated symbols received the same regime-routed strategies regardless of profile scoring. Scanner pipeline simplified from 7 stages to 5 stages: universe → technicals+validation → flow+ranking → watchlist. Strategies are now assigned purely via `strategy_routing` config per regime. ~200 lines of dead/bypassed code removed.

- **Archived Strategies** (`src/strategies/archived/`):
  - Moved `failed_breakout.py` to archive - 5-year backtest showed no profitable edge in any regime
  - Moved `trend_pullback.py` to archive - 5-year backtest showed -$1.8M loss with no profitable conditions
  - Removed from isolation config generator (`scripts/generate_isolation_configs.py`)

### Performance

- **Backtest Engine Optimizations** - ~7x speedup for large backtests:
  - **Order List Indexing** (`src/backtest/mock_executor.py`): Added `_pending_by_symbol` dict for O(1) order lookup per bar instead of O(N) scanning through all pending orders
  - **Sync Core Loop** (`src/backtest/runner.py`): Extracted synchronous `_process_loop_event_core()` from async wrapper to eliminate async overhead for CPU-bound bar processing
  - **Lazy Event Stream Merge** (`src/backtest/runner.py`): Replaced eager `list.sort()` with `heapq.merge()` for O(N) lazy merging of pre-sorted per-symbol bar streams
  - Verified: All 27 backtest tests pass; 5-year 25M bar backtest completes in ~8.5 minutes vs ~1 hour previously

### Documentation

- Added integration planning docs for Cerberus migration to Data-Gateway + Heber:
  - `docs/cerberus-data-gateway-heber-architecture.md`
  - `docs/cerberus-data-gateway-heber-migration-roadmap.md`
  - `docs/cerberus-data-gateway-heber-implementation-checklist.md`
- Added Data-Gateway/Heber runtime variable reference updates to:
  - `docs/environment-variables.md`
  - `.env.example`
- Comprehensive documentation audit and remediation completed.
- Rewrote `README.md` to match current runtime architecture, commands, and modules.
- Reworked `docs/architecture.md` with updated system/data-flow diagrams and module map.
- Added `docs/environment-variables.md` as source-of-truth env var reference.
- Updated `.env.example` to include current runtime vars and APCA aliases.
- Updated `docs/runbook.md`, `docs/order_flow.md`, and `docs/strategy_guide.md` for current interfaces/CLI behavior.
- Updated `CONTRIBUTING.md`, `TESTING.md`, and `SECURITY.md` for current workflows.
- Removed stale auto-generated `codebase.md`.
- **PRD.md Audit Update (Dec 2025)**: Comprehensive alignment of PRD with implementation
  - Added Section 12: Advanced Exit System (trailing stops, partial profits, regime-aware stops)
  - Added Section 13: Backtesting Engine (volume-aware fills, slippage modeling, ATR spreads)
  - Added strategies 9 (Momentum Continuation) and 10 (VIX Spike Fade) to Section 7.2
  - Marked PRD Regime Upgrade Patch as IMPLEMENTED
- **README.md**: Updated capabilities to include 5-axis regime system, advanced exits, and backtesting engine
- **architecture.md**: Updated Key Design Principles with 5-axis regime system replacing legacy routing

## 2026-01-05

### Fixed

- **Critical ConfigLoader Bug**: Fixed issue where specific config files (e.g., `config_vwap_trend_rider.yaml`) were being ignored - ConfigLoader now loads the specific file AFTER suite files to properly override settings
- **Strategy Isolation Configs**: Added ORB to `generate_isolation_configs.py` - now generates all 7 strategy configs

### Changed

- `src/core/config.py`: ConfigLoader.load_config() now tracks specific file paths and loads them after the suite files to allow proper overrides
- `scripts/generate_isolation_configs.py`: Added ORB strategy configuration with OPEN_ACTIVATION filters
