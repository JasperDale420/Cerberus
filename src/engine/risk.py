from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.analysis.momentum_crash import MomentumCrashDetector
from src.config.models import RiskConfig, StrategyConfig
from src.core.domain import MarketState, OrderIntent, OrderType, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.engine.cppi import CPPISizer
from src.engine.cvar_sizer import CVaRSizer
from src.engine.kelly import KellySizer


class RiskManager:
    """
    Enforces risk limits and converts Signals to OrderIntents.
    """

    DEFAULT_TIMEZONE = "US/Eastern"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        self.raw_config = config
        self.logger = logger

        # Load risk config using Pydantic
        # Prefer nested "risk" key, but support flat config for backward compat if needed
        risk_data = config.get("risk") if isinstance(config.get("risk"), dict) else config

        # Strategies config is often at root level in legacy configs, we might need to map it
        # However, new standard should be under risk or root.
        # Attempt to conform to model.
        try:
            # If strategies are at root, we might want to inject them into risk_data if not present
            if isinstance(risk_data, dict) and "strategies" not in risk_data and "strategies" in config:
                risk_data = risk_data.copy()
                risk_data["strategies"] = config["strategies"]

            self.risk_cfg = RiskConfig(**risk_data) if isinstance(risk_data, dict) else RiskConfig()
        except Exception as e:
            self.logger.error("Invalid Risk Configuration", error=str(e))
            # Fallback to defaults if critical failure, or re-raise?
            # Fail fast is better, but to stay safe we create a default config
            self.risk_cfg = RiskConfig()

        # Shortcuts from pydantic context
        self.max_daily_loss = self.risk_cfg.max_daily_loss
        self.max_risk_per_trade = self.risk_cfg.max_risk_per_trade
        self.max_open_risk = self.risk_cfg.max_open_risk
        self.max_trades_per_day = self.risk_cfg.max_trades_per_day
        self.max_trades_per_strategy = self.risk_cfg.max_trades_per_strategy
        self.risk_mode = self.risk_cfg.risk_mode.lower()

        # P2.1 fix: Now using Pydantic model field instead of raw config
        self.max_orders_per_day = self.risk_cfg.max_orders_per_day

        self.max_open_positions = self.risk_cfg.max_open_positions
        self.max_notional_per_order = self.risk_cfg.max_notional_per_order
        self.max_notional_per_symbol = self.risk_cfg.max_notional_per_symbol

        self.current_daily_pnl = 0.0
        self.daily_order_count = 0
        self.daily_entry_count = 0
        self.daily_completed_trade_count = 0
        self.per_strategy_entry_count: Dict[str, int] = {}
        self.per_strategy_completed_trade_count: Dict[str, int] = {}
        self._session_date: Optional[date] = None
        # M2 fix: Track positions carried from previous session
        self.positions_carried_forward = 0
        self.last_rejection_reason: Optional[str] = None

        # Kelly Criterion position sizer
        self.kelly_sizer = KellySizer(self.risk_cfg.kelly, self.logger)

        # CPPI drawdown-controlled sizer
        self.cppi_sizer = CPPISizer(config=self.risk_cfg.cppi, logger=self.logger)

        # HRP cross-strategy allocator
        from src.engine.hrp import HRPAllocator

        self.hrp_allocator = HRPAllocator(config=self.risk_cfg.hrp, logger=self.logger)

        # CVaR tail-risk sizer
        self.cvar_sizer = CVaRSizer(config=self.risk_cfg.cvar, logger=self.logger)

        # Momentum crash hedging detector (Daniel & Moskowitz 2016)
        from src.analysis.momentum_crash import MomentumCrashConfig as MCConfig

        mc_cfg = self.risk_cfg.momentum_crash
        self.momentum_crash_detector = MomentumCrashDetector(
            config=MCConfig(
                min_exposure=mc_cfg.min_exposure,
                bear_threshold=mc_cfg.bear_threshold,
                spread_lookback=mc_cfg.spread_lookback,
            ),
            logger=self.logger,
        )

    def _session_date_for(self, as_of: datetime) -> date:
        tz_name = str(self.raw_config.get("timezone", self.DEFAULT_TIMEZONE) or self.DEFAULT_TIMEZONE)
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo(self.DEFAULT_TIMEZONE)

        dt = as_of
        if isinstance(dt, datetime) and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(tz).date()

    def _maybe_rollover(self, as_of: Optional[datetime]) -> None:
        if not isinstance(as_of, datetime):
            return
        session_date = self._session_date_for(as_of)
        if self._session_date is None:
            self._session_date = session_date
            return
        if session_date == self._session_date:
            return

        # M2 fix: Capture position count BEFORE reset for accurate logging
        # positions_carried_forward was set by the last apply() call in previous session
        carried_count = self.positions_carried_forward
        # P2.2 fix: Capture old session date BEFORE updating
        old_session_date = self._session_date

        self._session_date = session_date
        self.current_daily_pnl = 0.0
        self.daily_order_count = 0
        self.daily_entry_count = 0
        self.daily_completed_trade_count = 0
        self.per_strategy_entry_count.clear()
        self.per_strategy_completed_trade_count.clear()
        self.positions_carried_forward = 0
        self.last_rejection_reason = None

        # Log after reset with captured values
        self.logger.info(
            "Session rollover",
            old_date=str(old_session_date),
            new_date=str(session_date),
            positions_carried_forward=carried_count,
        )

    def _get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        return self.risk_cfg.strategies.get(strategy_name)

    def _check_basic_gates(self, strat_cfg: Optional[StrategyConfig]) -> bool:
        if strat_cfg is not None and not strat_cfg.enabled:
            self.last_rejection_reason = "STRATEGY_DISABLED"
            return False

        if self.risk_mode == "off":
            self.last_rejection_reason = "RISK_MODE_OFF"
            return False

        return True

    def _check_volume_gates(self, signal: Signal) -> bool:
        if self.max_trades_per_day > 0 and self.daily_entry_count >= self.max_trades_per_day:
            self.last_rejection_reason = "MAX_TRADES_PER_DAY"
            return False

        if self.max_trades_per_strategy > 0:
            strat_entries = self.per_strategy_entry_count.get(signal.strategy, 0)
            if strat_entries >= self.max_trades_per_strategy:
                self.last_rejection_reason = "MAX_STRAT_TRADES"
                return False

        if self.daily_order_count >= self.max_orders_per_day:
            self.last_rejection_reason = "MAX_DAILY_ORDERS"
            return False

        return True

    def _check_position_gates(
        self,
        signal: Signal,
        symbol_state: SymbolState,
        current_positions: Optional[List[Any]],
    ) -> bool:
        if current_positions is None or symbol_state.position is not None:
            return True

        if len(current_positions) >= self.max_open_positions:
            self.last_rejection_reason = "MAX_POSITIONS"
            return False

        max_strat_pos = self.risk_cfg.max_positions_per_strategy
        strat_positions = [p for p in current_positions if getattr(p, "strategy", "") == signal.strategy]

        if len(strat_positions) >= max_strat_pos:
            self.last_rejection_reason = "MAX_STRAT_POSITIONS"
            return False

        return True

    def _check_loss_gate(self, account_equity: float = 0.0) -> bool:
        # Compute pct-based limit, then take the smaller of pct-based and dollar ceiling
        pct_limit = account_equity * self.risk_cfg.daily_loss_pct if account_equity > 0 else 0.0
        effective_limit = min(pct_limit, self.max_daily_loss) if pct_limit > 0 else self.max_daily_loss
        if self.current_daily_pnl <= -effective_limit:
            self.last_rejection_reason = "MAX_DAILY_LOSS"
            return False
        return True

    def _apply_risk_mode(self, effective_max_risk: float) -> float:
        if self.risk_mode == "reduced":
            return effective_max_risk * 0.5
        return effective_max_risk

    def _compute_base_risk(self, account_equity: float) -> float:
        """Compute base risk per trade from equity percentage, capped by dollar ceiling."""
        if account_equity > 0 and self.risk_cfg.risk_pct > 0:
            pct_risk = account_equity * self.risk_cfg.risk_pct
            return min(pct_risk, self.max_risk_per_trade)
        return self.max_risk_per_trade

    def _get_regime_multiplier(self, market_state: MarketState, strategy_name: Optional[str] = None) -> float:
        """
        Compute combined regime risk multiplier from multi-axis regime snapshot.

        Returns product of vol, liquidity, and risk axis multipliers.
        Returns 1.0 if no regime snapshot available.

        If strategy_name is provided and the strategy has a non-linear convexity class,
        antifragile per-class overrides are applied instead of the global defaults.
        """
        snapshot = market_state.regime_snapshot
        if snapshot is None:
            return 1.0

        multipliers = self.risk_cfg.regime_risk_multipliers

        # Get multiplier for each axis (default to 1.0 if not configured)
        vol_mult = multipliers.get("vol", {}).get(snapshot.vol.value, 1.0)
        liq_mult = multipliers.get("liquidity", {}).get(snapshot.liquidity.value, 1.0)
        # Risk axis is only meaningful if backed by a volatility proxy (e.g., VXX).
        # If we don't have a configured vol symbol, keep risk multiplier neutral.
        if getattr(snapshot, "vol_symbol", None):
            risk_mult = multipliers.get("risk", {}).get(snapshot.risk.value, 1.0)
        else:
            risk_mult = 1.0

        # Complexity axis from entropy overlay
        complexity_mult = 1.0
        if snapshot.complexity is not None:
            complexity_mult = multipliers.get("complexity", {}).get(snapshot.complexity.value, 1.0)

        combined = vol_mult * liq_mult * risk_mult * complexity_mult

        # Antifragile: apply per-class overrides if strategy has a convexity class
        if strategy_name:
            convexity = self._get_convexity_class(strategy_name)
            if convexity != "linear":
                combined = self._apply_antifragile_overrides(combined, snapshot, convexity)

        return max(combined, 0.0)

    def _get_convexity_class(self, strategy_name: str) -> str:
        """Get the convexity class for a strategy. Defaults to 'linear'."""
        strat_cfg = self._get_strategy_config(strategy_name)
        if strat_cfg is None:
            return "linear"
        return getattr(strat_cfg, "convexity_class", "linear")

    def _apply_antifragile_overrides(self, base_combined: float, snapshot, convexity: str) -> float:
        """Apply antifragile per-class regime overrides to the base multiplier."""
        overrides = self.risk_cfg.antifragile_overrides.get(convexity)
        if not overrides:
            return base_combined

        # Compute the override multiplier from vol and risk axes
        override_mult = 1.0

        vol_overrides = overrides.get("vol", {})
        if vol_overrides:
            override_mult *= vol_overrides.get(snapshot.vol.value, 1.0)

        risk_overrides = overrides.get("risk", {})
        if risk_overrides and getattr(snapshot, "vol_symbol", None):
            override_mult *= risk_overrides.get(snapshot.risk.value, 1.0)

        return max(override_mult, 0.0)

    def _apply_strategy_limits(
        self,
        effective_max_risk: float,
        signal: Signal,
        strat_cfg: Optional[StrategyConfig],
    ) -> float:
        if strat_cfg is None:
            return effective_max_risk

        if strat_cfg.max_risk_per_trade is not None:
            effective_max_risk = min(effective_max_risk, strat_cfg.max_risk_per_trade)

        # Backward-compatible legacy regime override support.
        # Prefer explicit signal.regime if provided by older callers/tests.
        if strat_cfg.regimes:
            regime_name: Optional[str] = None
            if signal.regime is not None:
                regime_name = str(getattr(signal.regime, "value", signal.regime))
            elif isinstance(signal.regime_tags, dict):
                legacy = signal.regime_tags.get("legacy")
                if legacy:
                    regime_name = str(legacy)

            if regime_name:
                r_cfg = strat_cfg.regimes.get(regime_name)
                if r_cfg is not None:
                    if not r_cfg.enabled:
                        self.last_rejection_reason = "REGIME_DISABLED"
                        return 0.0
                    if r_cfg.max_risk_per_trade is not None:
                        effective_max_risk = min(effective_max_risk, r_cfg.max_risk_per_trade)

        return effective_max_risk

    def _apply_equity_constraints(
        self,
        qty_limit: int,
        signal: Signal,
        account_equity: float,
        risk_per_share: float,
        effective_max_risk: float,
    ) -> int:
        if account_equity > 0 and signal.entry_price > 0:
            # Use Kelly-computed fraction if available, otherwise fall back to static config
            kelly_frac = self.kelly_sizer.get_kelly_fraction(signal.strategy)
            if kelly_frac is not None:
                max_equity_pct = kelly_frac
            else:
                max_equity_pct = getattr(self.risk_cfg, "max_equity_pct", 0.10)

            max_equity_allocation = account_equity * max_equity_pct
            max_qty_equity = int(max_equity_allocation / signal.entry_price)
            if max_qty_equity < qty_limit:
                self.logger.info(
                    "Risk sizing constrained by equity",
                    symbol=signal.symbol,
                    equity=account_equity,
                    max_equity_pct=round(max_equity_pct, 4),
                    kelly_active=kelly_frac is not None,
                    max_equity_qty=max_qty_equity,
                    risk_qty=int(effective_max_risk / risk_per_share),
                )
                return max_qty_equity
        return qty_limit

    def _calculate_qty(
        self,
        signal: Signal,
        account_equity: float,
        strat_cfg: Optional[StrategyConfig],
        market_state: Optional[MarketState] = None,
        symbol_state: Optional[SymbolState] = None,
    ) -> int:
        # Check if this is a pair trade leg
        if signal.meta.get("pair_trade"):
            effective_max_risk = self._compute_base_risk(account_equity)
            effective_max_risk = self._apply_risk_mode(effective_max_risk)
            effective_max_risk = self._apply_strategy_limits(effective_max_risk, signal, strat_cfg)
            return self._calculate_pair_qty(signal, account_equity)

        risk_per_share = abs(signal.entry_price - signal.stop_price)
        if risk_per_share <= 0:
            self.last_rejection_reason = "INVALID_STOP"
            return 0

        effective_max_risk = self._compute_base_risk(account_equity)
        effective_max_risk = self._apply_risk_mode(effective_max_risk)
        effective_max_risk = self._apply_strategy_limits(effective_max_risk, signal, strat_cfg)

        # Apply Tournament-style Rank-based scaling
        rank_mult = self._get_alpha_rank_multiplier(signal, symbol_state)
        if abs(rank_mult - 1.0) > 1e-6:
            effective_max_risk *= rank_mult
            self.logger.debug(
                "Risk scaled by alpha rank",
                symbol=signal.symbol,
                rank_mult=rank_mult,
                scaled_risk=effective_max_risk,
            )

        if effective_max_risk <= 0:
            if self.last_rejection_reason is None:
                self.last_rejection_reason = "ZERO_RISK_LIMIT"
            return 0

        qty_limit = int(effective_max_risk / risk_per_share)
        qty_limit = self._apply_equity_constraints(
            qty_limit, signal, account_equity, risk_per_share, effective_max_risk
        )

        qty: int = 0
        if signal.size_hint:
            if signal.size_hint <= 1.0:
                # size_hint is a conviction multiplier (0-1) — scale qty_limit
                qty = max(int(qty_limit * signal.size_hint), 1) if qty_limit > 0 else 0
            else:
                # size_hint is an absolute share count
                qty = min(int(signal.size_hint), qty_limit)
        else:
            qty = qty_limit

        if qty <= 0:
            if self.last_rejection_reason is None:
                self.last_rejection_reason = "ZERO_QTY"
            return 0

        # PRD Addendum: Apply regime-based risk multiplier
        if market_state is not None:
            regime_mult = self._get_regime_multiplier(market_state, strategy_name=signal.strategy)
            if regime_mult < 1e-9:
                self.last_rejection_reason = "REGIME_RISK_ZERO"
                self.logger.info(
                    "Signal rejected: Regime risk multiplier is zero",
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    regime_tags=getattr(market_state.regime_snapshot, "regime_tags", {}),
                )
                return 0
            if regime_mult < 1.0:
                orig_qty = qty
                # P0 fix: Calculate adjusted qty without forcing minimum 1
                # This prevents trades in high-risk regimes where sizing should be zero
                adjusted_qty = int(qty * regime_mult)
                if adjusted_qty < 1:
                    self.last_rejection_reason = "REGIME_RISK_BELOW_MIN"
                    self.logger.info(
                        "Signal rejected: Regime-adjusted qty below minimum",
                        symbol=signal.symbol,
                        strategy=signal.strategy,
                        orig_qty=orig_qty,
                        adjusted_qty=adjusted_qty,
                        regime_mult=round(regime_mult, 3),
                    )
                    return 0
                qty = adjusted_qty
                if qty != orig_qty:
                    self.logger.debug(
                        "Qty adjusted by regime multiplier",
                        symbol=signal.symbol,
                        orig_qty=orig_qty,
                        adjusted_qty=qty,
                        regime_mult=round(regime_mult, 3),
                    )

        # CPPI drawdown-controlled sizing
        if self.risk_cfg.cppi.enabled and qty > 0:
            cppi_mult = self.cppi_sizer.get_cppi_multiplier()
            if cppi_mult < 1.0:
                orig_qty = qty
                adjusted_qty = int(qty * cppi_mult)
                if adjusted_qty < 1:
                    self.last_rejection_reason = "CPPI_BELOW_MIN"
                    self.logger.info(
                        "Signal rejected: CPPI-adjusted qty below minimum",
                        symbol=signal.symbol,
                        strategy=signal.strategy,
                        orig_qty=orig_qty,
                        cppi_mult=round(cppi_mult, 3),
                    )
                    return 0
                qty = adjusted_qty
                if qty != orig_qty:
                    self.logger.debug(
                        "Qty adjusted by CPPI",
                        symbol=signal.symbol,
                        orig_qty=orig_qty,
                        adjusted_qty=qty,
                        cppi_mult=round(cppi_mult, 3),
                    )

        # HRP cross-strategy allocation scaling
        if self.risk_cfg.hrp.enabled and qty > 0:
            hrp_weight = self.hrp_allocator.get_strategy_weight(signal.strategy)
            # HRP weight represents capital allocation fraction relative to equal-weight
            # If N strategies, equal weight = 1/N. HRP weight adjusts this.
            # Scale qty proportionally: if HRP gives 2x the equal weight, double qty
            n_strategies = max(len(self.risk_cfg.strategies), 1)
            equal_weight = 1.0 / n_strategies
            if equal_weight > 0:
                hrp_scale = hrp_weight / equal_weight
                hrp_scale = max(0.5, min(2.0, hrp_scale))  # Clamp scaling factor
                if abs(hrp_scale - 1.0) > 0.01:
                    orig_qty = qty
                    qty = max(1, int(qty * hrp_scale))
                    if qty != orig_qty:
                        self.logger.debug(
                            "Qty adjusted by HRP allocation",
                            symbol=signal.symbol,
                            strategy=signal.strategy,
                            orig_qty=orig_qty,
                            adjusted_qty=qty,
                            hrp_weight=round(hrp_weight, 4),
                            hrp_scale=round(hrp_scale, 3),
                        )

        # CVaR tail-risk sizing
        if self.risk_cfg.cvar.enabled and qty > 0:
            cvar_mult = self.cvar_sizer.get_cvar_multiplier()
            if cvar_mult < 1.0:
                orig_qty = qty
                adjusted_qty = int(qty * cvar_mult)
                if adjusted_qty < 1:
                    self.last_rejection_reason = "CVAR_BELOW_MIN"
                    self.logger.info(
                        "Signal rejected: CVaR-adjusted qty below minimum",
                        symbol=signal.symbol,
                        strategy=signal.strategy,
                        orig_qty=orig_qty,
                        cvar_mult=round(cvar_mult, 3),
                    )
                    return 0
                qty = adjusted_qty
                if qty != orig_qty:
                    self.logger.debug(
                        "Qty adjusted by CVaR tail risk",
                        symbol=signal.symbol,
                        orig_qty=orig_qty,
                        adjusted_qty=qty,
                        cvar_mult=round(cvar_mult, 3),
                    )

        # Momentum crash hedging (Daniel & Moskowitz 2016)
        if self.risk_cfg.momentum_crash.enabled and qty > 0:
            crash_result = self.momentum_crash_detector.get_crash_probability()
            if crash_result is not None and crash_result.exposure_scalar < 1.0:
                orig_qty = qty
                adjusted_qty = int(qty * crash_result.exposure_scalar)
                if adjusted_qty < 1:
                    self.last_rejection_reason = "MOMENTUM_CRASH_BELOW_MIN"
                    self.logger.info(
                        "Signal rejected: Momentum crash-adjusted qty below minimum",
                        symbol=signal.symbol,
                        strategy=signal.strategy,
                        orig_qty=orig_qty,
                        exposure_scalar=round(crash_result.exposure_scalar, 3),
                    )
                    return 0
                qty = adjusted_qty
                if qty != orig_qty:
                    self.logger.debug(
                        "Qty adjusted by momentum crash hedging",
                        symbol=signal.symbol,
                        orig_qty=orig_qty,
                        adjusted_qty=qty,
                        crash_prob=round(crash_result.crash_probability, 4),
                        exposure_scalar=round(crash_result.exposure_scalar, 3),
                    )

        return qty

    def _calculate_pair_qty(self, signal: Signal, account_equity: float) -> int:
        """
        Calculates quantity for a pair trade leg.
        Ensures legs are balanced based on the hedge ratio.
        """
        pair_info = signal.meta.get("pair_trade")
        if not pair_info:
            return 0

        hedge_ratio = pair_info.get("hedge_ratio", 1.0)

        # Target Notional for reference leg = 1% of equity.
        # StatArb usually targets notional neutrality.
        target_notional_ref = account_equity * 0.01

        if abs(hedge_ratio - 1.0) < 1e-6:
            # Reference leg
            qty = int(target_notional_ref / signal.entry_price)
        else:
            # Dependent leg (s1 = h * s2)
            partner_price = pair_info.get("partner_price", signal.entry_price)
            qty_ref = int(target_notional_ref / partner_price)
            qty = int(abs(hedge_ratio) * qty_ref)

        self.logger.info(
            "Pair position sized",
            symbol=signal.symbol,
            qty=qty,
            hedge_ratio=hedge_ratio,
            pair_id=pair_info.get("pair_id"),
        )
        return max(1, qty)

    def _get_alpha_rank_multiplier(self, signal: Signal, symbol_state: Optional[SymbolState] = None) -> float:
        """
        Fetch alpha_rank from signal or symbol metadata and return corresponding multiplier.
        """
        rank = signal.meta.get("alpha_rank")
        if rank is None and symbol_state is not None:
            rank = symbol_state.meta.get("alpha_rank")

        if rank is None:
            return 1.0

        try:
            rank_int = int(rank)
        except (ValueError, TypeError):
            return 1.0

        if rank_int <= 0:
            return 1.0

        multipliers = self.risk_cfg.alpha_rank_multipliers
        if not multipliers:
            return 1.0

        # Find the best matching rank (e.g. if rank is 4, and we have {3: 1.1, 5: 1.0}, use 5)
        # We search for the smallest rank in config that is >= our rank
        sorted_ranks = sorted(multipliers.keys())
        target_mult = 1.0

        # Exact match or find the next tier
        if rank_int in multipliers:
            return float(multipliers[rank_int])

        for r in sorted_ranks:
            if r >= rank_int:
                target_mult = float(multipliers[r])
                break
        else:
            # If rank is higher than all tiers (e.g. rank 20, max tier 10), use the last tier
            target_mult = float(multipliers[sorted_ranks[-1]])

        return target_mult

    def _get_effective_notional_limit(self, account_equity: float = 0.0) -> float:
        """Calculate the effective per-order notional limit."""
        max_notional_pct = getattr(self.risk_cfg, "max_notional_pct", 0.50)
        if account_equity > 0 and max_notional_pct > 0:
            return account_equity * max_notional_pct
        elif self.max_notional_per_order > 0:
            return self.max_notional_per_order
        return float("inf")

    def _check_symbol_notional(self, notional: float, symbol_state: SymbolState) -> bool:
        """Check per-symbol notional limit (existing + proposed)."""
        if self.max_notional_per_symbol > 0:
            existing_notional = 0.0
            if symbol_state.position:
                existing_notional = float(symbol_state.position.qty * symbol_state.position.avg_price)
            if (existing_notional + notional) > self.max_notional_per_symbol:
                self.last_rejection_reason = "MAX_SYMBOL_NOTIONAL"
                return False
        return True

    def _check_notional_limits(self, notional: float, symbol_state: SymbolState, account_equity: float = 0.0) -> bool:
        """Legacy method — kept for backward compat. New flow uses cap-then-check."""
        effective_limit = self._get_effective_notional_limit(account_equity)
        if notional > effective_limit:
            self.logger.warning(
                "Signal rejected: Max notional per order exceeded",
                symbol=symbol_state.symbol,
                notional=round(notional, 2),
                limit=round(effective_limit, 2),
                reason_code="MAX_NOTIONAL",
            )
            self.last_rejection_reason = "MAX_NOTIONAL"
            return False
        return self._check_symbol_notional(notional, symbol_state)

    def _check_risk_exposure(
        self, risk_per_share: float, qty: int, current_positions: Optional[List[Any]], account_equity: float = 0.0
    ) -> bool:
        # Compute pct-based open risk limit, capped by dollar ceiling
        pct_limit = account_equity * self.risk_cfg.open_risk_pct if account_equity > 0 else 0.0
        effective_limit = min(pct_limit, self.max_open_risk) if pct_limit > 0 else self.max_open_risk
        if effective_limit > 0 and current_positions is not None:
            open_risk = 0.0
            for p in current_positions:
                open_risk += float(getattr(p, "open_risk", 0.0) or 0.0)
            proposed_risk = risk_per_share * qty
            if (open_risk + proposed_risk) > effective_limit:
                self.last_rejection_reason = "MAX_OPEN_RISK"
                return False
        return True

    def _create_intent(self, signal: Signal, qty: int) -> OrderIntent:
        time_in_force = self.risk_cfg.time_in_force
        return OrderIntent(
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            order_type=OrderType.LIMIT,
            limit_price=signal.entry_price,
            time_in_force=time_in_force,
            correlation_id=signal.correlation_id,
            stop_loss=signal.stop_price,
            take_profit=signal.target_price,
            strategy=signal.strategy,
            meta={"created_at": signal.generated_at.isoformat()},
        )

    def apply(
        self,
        signal: Signal,
        symbol_state: SymbolState,
        market_state: MarketState,
        current_positions: Optional[List[Any]] = None,
        account_equity: float = 0.0,
    ) -> List[OrderIntent]:
        """
        Evaluates a signal and returns a list of OrderIntents if approved, or an empty list if rejected.
        """
        # M2 fix: Track current position count for rollover observability
        if current_positions is not None:
            self.positions_carried_forward = len(current_positions)

        self._maybe_rollover(getattr(market_state, "time", None))
        self.last_rejection_reason = None

        strat_cfg = self._get_strategy_config(signal.strategy)

        if not self._check_basic_gates(strat_cfg):
            self._log_rejection(signal)
            return []

        # Legacy regime-specific enable check removed
        # Multi-axis regime gating is handled at strategy engine level

        if not self._check_volume_gates(signal):
            self._log_rejection(signal)
            return []

        if not self._check_position_gates(signal, symbol_state, current_positions):
            self._log_rejection(signal)
            return []

        if not self._check_loss_gate(account_equity):
            self._log_rejection(signal)
            return []

        # Calculate Qty
        qty = self._calculate_qty(signal, account_equity, strat_cfg, market_state, symbol_state)
        if qty <= 0:
            self._log_rejection(signal)
            return []

        # Cap qty to fit within notional limit (instead of rejecting)
        notional = qty * signal.entry_price
        max_notional = self._get_effective_notional_limit(account_equity)
        if max_notional > 0 and notional > max_notional:
            capped_qty = int(max_notional / signal.entry_price)
            if capped_qty <= 0:
                self.last_rejection_reason = "MAX_NOTIONAL"
                self._log_rejection(signal)
                return []
            self.logger.debug(
                "Notional cap: reducing qty",
                symbol=signal.symbol,
                orig_qty=qty,
                capped_qty=capped_qty,
                notional=round(qty * signal.entry_price, 2),
                limit=round(max_notional, 2),
            )
            qty = capped_qty
        # Also check per-symbol notional
        if not self._check_symbol_notional(qty * signal.entry_price, symbol_state):
            self._log_rejection(signal)
            return []

        risk_per_share = abs(signal.entry_price - signal.stop_price)
        if not self._check_risk_exposure(risk_per_share, qty, current_positions, account_equity):
            self._log_rejection(signal)
            return []

        intent = self._create_intent(signal, qty)

        self.daily_order_count += 1
        self.daily_entry_count += 1
        self.per_strategy_entry_count[signal.strategy] = self.per_strategy_entry_count.get(signal.strategy, 0) + 1
        self.logger.info(
            "Signal approved",
            symbol=intent.symbol,
            strategy=intent.strategy,
            correlation_id=intent.correlation_id,
            regime_tags=signal.regime_tags,
            intent=intent,
            daily_order_count=self.daily_order_count,
            daily_entry_count=self.daily_entry_count,
        )
        return [intent]

    def _log_rejection(self, signal: Signal) -> None:
        self.logger.warning(
            f"Signal rejected: {self.last_rejection_reason}",
            symbol=signal.symbol,
            strategy=signal.strategy,
            correlation_id=getattr(signal, "correlation_id", "") or None,
            regime_tags=signal.regime_tags,
            reason_code=self.last_rejection_reason,
        )

    def update_pnl(
        self, pnl: float, *, as_of: Optional[datetime] = None, account_equity: Optional[float] = None
    ) -> None:
        """
        Updates the current daily PnL and CPPI equity tracker.
        """
        self._maybe_rollover(as_of)
        self.current_daily_pnl += pnl

        # Update CPPI with current equity if provided
        if self.risk_cfg.cppi.enabled and account_equity is not None:
            self.cppi_sizer.update_equity(account_equity)

    def update_momentum_crash(
        self,
        realized_vol: float,
        is_bear: bool,
        credit_spread_zscore: float = 0.0,
    ) -> None:
        """Feed market conditions to momentum crash detector."""
        if self.risk_cfg.momentum_crash.enabled:
            self.momentum_crash_detector.update(
                realized_vol=realized_vol,
                is_bear=is_bear,
                credit_spread_zscore=credit_spread_zscore,
            )

    def record_completed_trade(
        self,
        strategy: str,
        *,
        pnl_net: Optional[float] = None,
        as_of: Optional[datetime] = None,
    ) -> None:
        """Increment completed-trade counters and feed Kelly sizer."""
        self._maybe_rollover(as_of)
        self.daily_completed_trade_count += 1
        self.per_strategy_completed_trade_count[strategy] = self.per_strategy_completed_trade_count.get(strategy, 0) + 1

        # Feed trade result to Kelly sizer for dynamic position sizing
        if pnl_net is not None:
            self.kelly_sizer.record_trade(strategy, pnl_net)

        # Feed daily return to HRP for cross-strategy allocation
        if self.risk_cfg.hrp.enabled and pnl_net is not None:
            from datetime import date as date_type

            if as_of is None:
                trade_date = date_type.today()
            elif isinstance(as_of, (int, float)):
                from datetime import datetime as dt_type
                from datetime import timezone

                as_of_dt = (
                    dt_type.fromtimestamp(as_of / 1000, tz=timezone.utc)
                    if as_of > 1e10
                    else dt_type.fromtimestamp(as_of, tz=timezone.utc)
                )
                trade_date = as_of_dt.date()
            else:
                trade_date = as_of.date()
            # Approximate daily return from PnL (HRP accumulates these)
            self.hrp_allocator.record_daily_return(strategy, trade_date, pnl_net)
