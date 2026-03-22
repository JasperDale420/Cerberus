from __future__ import annotations

from collections import deque
from typing import Any

from src.analysis.ou_estimator import OUEstimator
from src.core.domain import (
    Bar,
    MarketState,
    OrderSide,
    Signal,
    SymbolState,
    TrendRegime,
    VolRegime,
)
from src.core.indicators import RollingStd
from src.core.logger import StructuredLogger
from src.quant.cointegration import RollingCointegrationMonitor
from src.quant.filters import KalmanHedgeRatio
from src.strategies.base import BaseStrategy
from src.strategies.confluence import ConfluenceScorer, score_deviation, score_volume


def _pair_key(leg_a: str, leg_b: str) -> str:
    """Canonical key for a pair (alphabetically sorted)."""
    a, b = sorted((leg_a, leg_b))
    return f"{a}:{b}"


def _make_pair_state(
    leg_a: str,
    leg_b: str,
    spread_lookback: int,
) -> dict[str, Any]:
    return {
        "leg_a": leg_a,
        "leg_b": leg_b,
        "kalman": KalmanHedgeRatio(),
        "spread_std": RollingStd.create(spread_lookback),
        "last_price_a": 0.0,
        "last_price_b": 0.0,
        "bar_count": 0,
        "bar_idx_sync": -1,  # synchronized bar index for factor cache lookup
        "signal_active": False,
        "z_score_history": deque(maxlen=30),
        # Rolling correlation tracking
        "prices_a": deque(maxlen=spread_lookback),
        "prices_b": deque(maxlen=spread_lookback),
    }


class PairTradingV2Strategy(BaseStrategy):
    """Daily swing pair trading with Kalman-filtered hedge ratio,
    Engle-Granger cointegration gate, OU half-life gate, and rolling
    correlation monitoring.

    Designed for daily bars with multi-day holds.  ``on_bar`` is called
    per-symbol; the strategy tracks which pairs each symbol belongs to
    and evaluates signals when both legs have received a daily bar.
    """

    name: str = "pair_trading_v2"
    allow_overnight: bool = True

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)
        self._pair_state: dict[str, dict[str, Any]] = {}
        self._symbol_to_pairs: dict[str, list[str]] = {}
        # Quant monitors — keyed by pair key
        self._coint_monitors: dict[str, RollingCointegrationMonitor] = {}
        self._ou_estimators: dict[str, OUEstimator] = {}
        # Optional precomputed factor cache (injected during WFO optimization)
        self._factor_cache = config.get("_factor_cache", None)
        self._init_pairs(config)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.confluence_threshold = float(config.get("confluence_threshold", 60.0))
        self.entry_z_threshold = float(config.get("entry_z_threshold", 2.0))
        self.stop_z_threshold = float(config.get("stop_z_threshold", 3.5))
        self.spread_lookback = int(config.get("spread_lookback", 90))
        self.min_bars = int(config.get("min_bars", 30))
        self.max_hold_days = int(config.get("max_hold_days", 15))
        self.min_correlation = float(config.get("min_correlation", 0.6))
        self.tf_alignment_mode = "mean_reversion"

    def _init_pairs(self, config: dict[str, Any]) -> None:
        """Build internal pair state from config ``pairs`` list."""
        for pair in config.get("pairs", []):
            leg_a, leg_b = pair["leg_a"], pair["leg_b"]
            key = _pair_key(leg_a, leg_b)
            if key in self._pair_state:
                continue
            self._pair_state[key] = _make_pair_state(
                leg_a,
                leg_b,
                self.spread_lookback,
            )
            self._symbol_to_pairs.setdefault(leg_a, []).append(key)
            self._symbol_to_pairs.setdefault(leg_b, []).append(key)
            # Init quant monitors per pair
            self._coint_monitors[key] = RollingCointegrationMonitor(
                lookback=self.spread_lookback,
                retest_interval=20,
            )
            self._ou_estimators[key] = OUEstimator(
                lookback=self.spread_lookback,
                min_observations=30,
            )

    # -- regime helpers ---------------------------------------------------

    @staticmethod
    def _regime_ok(market_state: MarketState) -> bool:
        snap = market_state.regime_snapshot
        if snap is None:
            return True
        return snap.vol != VolRegime.SHOCK

    @staticmethod
    def _regime_flatness_score(market_state: MarketState) -> float:
        snap = market_state.regime_snapshot
        if snap is None:
            return 50.0
        if snap.trend == TrendRegime.FLAT:
            return 90.0
        if snap.trend in (TrendRegime.UP, TrendRegime.DOWN):
            return 50.0
        return 0.0

    # -- synchronization ---------------------------------------------------

    @staticmethod
    def _both_legs_have_prices(ps: dict[str, Any]) -> bool:
        """Both legs have received at least one bar (daily resolution)."""
        return ps["last_price_a"] > 0.0 and ps["last_price_b"] > 0.0

    # -- rolling correlation -----------------------------------------------

    @staticmethod
    def _rolling_correlation(ps: dict[str, Any]) -> float | None:
        """Compute rolling Pearson correlation from stored price deques."""
        prices_a = ps["prices_a"]
        prices_b = ps["prices_b"]
        n = len(prices_a)
        if n < 20:
            return None
        a_list = list(prices_a)
        b_list = list(prices_b)
        mean_a = sum(a_list) / n
        mean_b = sum(b_list) / n
        cov = sum((a_list[i] - mean_a) * (b_list[i] - mean_b) for i in range(n)) / n
        std_a = (sum((x - mean_a) ** 2 for x in a_list) / n) ** 0.5
        std_b = (sum((x - mean_b) ** 2 for x in b_list) / n) ** 0.5
        if std_a < 1e-12 or std_b < 1e-12:
            return None
        return cov / (std_a * std_b)

    # -- spread / z-score -------------------------------------------------

    def _compute_spread_zscore(
        self,
        ps: dict[str, Any],
        key: str,
    ) -> tuple[float, float, float, float] | None:
        """Return (hedge_ratio, spread, z_score, std) or None.

        When a ``PairFactorCache`` is available (WFO optimization), reads
        precomputed Kalman factors at the synchronized bar index instead of
        running the online Kalman filter.  The rolling std z-score is still
        computed online because it depends on ``spread_lookback`` (a tunable
        parameter).
        """
        price_a, price_b = ps["last_price_a"], ps["last_price_b"]
        if price_b <= 0.0 or price_a <= 0.0:
            return None

        # Try precomputed cache first (WFO mode)
        if self._factor_cache is not None:
            bar_idx = ps["bar_idx_sync"]
            factors = self._factor_cache.get(key, bar_idx)
            if factors is not None:
                spread = factors.spread
                hedge_ratio = factors.hedge_ratio
                # Still update spread_std for z-score (parameter-dependent)
                mean, std = ps["spread_std"].update(spread)
                if std <= 0.0:
                    return None
                z_score = (spread - mean) / std
                return hedge_ratio, spread, z_score, std

        # Fallback: online Kalman computation (live trading / no cache)
        kalman: KalmanHedgeRatio = ps["kalman"]
        spread = kalman.update(price_b, price_a)  # x=price_b, y=price_a
        hedge_ratio = kalman.hedge_ratio

        mean, std = ps["spread_std"].update(spread)

        if std <= 0.0:
            return None

        z_score = (spread - mean) / std

        return hedge_ratio, spread, z_score, std

    # -- confluence scoring -----------------------------------------------

    def _score_entry(
        self,
        z_score: float,
        ps: dict[str, Any],
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> ConfluenceScorer:
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)
        z_abs = abs(z_score)

        # 1. Spread z-score (0.35)
        scorer.add_factor(
            "spread_z_score",
            raw_value=z_score,
            score=score_deviation(z_abs, 2.0, 4.0),
            weight=0.35,
            passed=z_abs >= self.entry_z_threshold,
        )

        # 2. Spread stationarity proxy (0.25)
        history = ps["z_score_history"]
        stat_score = 50.0
        if len(history) >= 5:
            stat_score = 70.0 if z_abs > abs(history[-5]) else 50.0
        scorer.add_factor(
            "spread_stationarity",
            raw_value=stat_score,
            score=stat_score,
            weight=0.25,
        )

        # 3. Volume (0.15)
        avg_vol = float(symbol_state.meta.get("avg_volume_20", 0) or 0)
        vol_sc = score_volume(bar.volume, avg_vol, 1.0, 2.5) if avg_vol > 0 else 50.0
        scorer.add_factor(
            "volume",
            raw_value=bar.volume,
            score=vol_sc,
            weight=0.15,
            passed=vol_sc > 0.0,
        )

        # 4. Regime flatness (0.15)
        flat_sc = self._regime_flatness_score(market_state)
        scorer.add_factor(
            "regime_flatness",
            raw_value=flat_sc,
            score=flat_sc,
            weight=0.15,
            passed=flat_sc >= 50.0,
        )

        # 5. Hedge ratio stability via Kalman uncertainty (0.10)
        kalman: KalmanHedgeRatio = ps["kalman"]
        uncertainty = kalman.hedge_ratio_uncertainty
        # Lower uncertainty = more stable. Map [0, 10] -> [100, 0]
        h_sc = max(0.0, min(100.0, 100.0 * (1.0 - uncertainty / 10.0)))
        scorer.add_factor("hedge_stability", raw_value=h_sc, score=h_sc, weight=0.10)

        return scorer

    # -- stop / target ----------------------------------------------------

    def _compute_stop_target(
        self,
        side: OrderSide,
        bar: Bar,
        symbol_state: SymbolState,
        z_score: float,
        std: float,
        hedge_ratio: float,
    ) -> tuple[float, float]:
        price = bar.close
        hr_abs = abs(hedge_ratio) if abs(hedge_ratio) > 1e-4 else 1.0
        spread_std_price = std / hr_abs
        atr = float(symbol_state.meta.get("atr", price * 0.02) or price * 0.02)

        stop_dist = max(
            (self.stop_z_threshold - self.entry_z_threshold) * spread_std_price,
            2.0 * atr,
        )
        target_dist = abs(z_score) * spread_std_price

        if side == OrderSide.BUY:
            return price - stop_dist, price + target_dist
        return price + stop_dist, price - target_dist

    # -- side determination -----------------------------------------------

    def _determine_side(
        self,
        z_score: float,
        symbol: str,
        leg_a: str,
        leg_b: str,
    ) -> OrderSide | None:
        """z < -threshold => BUY leg_a / SELL leg_b (spread too low).
        z >  threshold => SELL leg_a / BUY leg_b (spread too high)."""
        if z_score < -self.entry_z_threshold:
            if symbol == leg_a:
                return OrderSide.BUY
            if symbol == leg_b:
                return OrderSide.SELL
        elif z_score > self.entry_z_threshold:
            if symbol == leg_a:
                return OrderSide.SELL
            if symbol == leg_b:
                return OrderSide.BUY
        return None

    # -- on_bar -----------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        pair_keys = self._symbol_to_pairs.get(symbol)
        if not pair_keys:
            return None
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._regime_ok(market_state):
            return None

        for key in pair_keys:
            signal = self._process_pair(key, symbol, bar, symbol_state, market_state)
            if signal is not None:
                return signal
        return None

    # -- per-pair processing ----------------------------------------------

    def _process_pair(
        self,
        key: str,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        ps = self._pair_state[key]
        leg_a, leg_b = ps["leg_a"], ps["leg_b"]

        # Update the leg that just received a bar
        if symbol == leg_a:
            ps["last_price_a"] = bar.close
        elif symbol == leg_b:
            ps["last_price_b"] = bar.close
        else:
            return None

        # Only count bars when both legs have prices.
        # With daily bars both legs get one bar per day.
        if self._both_legs_have_prices(ps):
            ps["bar_count"] += 1
            ps["bar_idx_sync"] += 1
            ps["prices_a"].append(ps["last_price_a"])
            ps["prices_b"].append(ps["last_price_b"])
            self._coint_monitors[key].update(ps["last_price_a"], ps["last_price_b"])

        # Warm-up phase: feed indicators but skip signal generation
        if ps["bar_count"] < self.min_bars:
            if ps["last_price_a"] > 0 and ps["last_price_b"] > 0:
                self._compute_spread_zscore(ps, key)
            return None

        if not self._both_legs_have_prices(ps):
            return None
        if ps["signal_active"]:
            return None

        result = self._compute_spread_zscore(ps, key)
        if result is None:
            return None
        hedge_ratio, spread, z_score, std = result

        ps["z_score_history"].append(z_score)

        # -- Quant gates (applied BEFORE confluence scoring) ----------------

        # Gate 1: Cointegration check
        if not self._coint_monitors[key].is_valid:
            self.logger.debug(
                "pair_trading_v2: cointegration gate rejected",
                pair=key,
                symbol=symbol,
            )
            return None

        # Gate 2: OU half-life check (units = trading days)
        ou_result = self._ou_estimators[key].update(spread)
        half_life_days = ou_result.half_life if ou_result is not None else 0.0
        if ou_result is not None and half_life_days > self.max_hold_days:
            self.logger.debug(
                "pair_trading_v2: OU half-life gate rejected",
                pair=key,
                symbol=symbol,
                half_life_days=round(half_life_days, 2),
                max_hold_days=self.max_hold_days,
            )
            return None

        # Track rolling correlation (informational; can gate exits later)
        corr = self._rolling_correlation(ps)

        # -- end quant gates ------------------------------------------------

        side = self._determine_side(z_score, symbol, leg_a, leg_b)
        if side is None:
            return None

        scorer = self._score_entry(z_score, ps, bar, symbol_state, market_state)
        if not scorer.passes_threshold():
            self.logger.debug(
                "pair_trading_v2: below confluence threshold",
                symbol=symbol,
                pair=key,
                score=round(scorer.score(), 2),
                z_score=round(z_score, 4),
            )
            return None

        stop_price, target_price = self._compute_stop_target(
            side,
            bar,
            symbol_state,
            z_score,
            std,
            hedge_ratio,
        )
        if side == OrderSide.BUY and stop_price >= bar.close:
            return None
        if side == OrderSide.SELL and stop_price <= bar.close:
            return None

        other_leg = leg_b if symbol == leg_a else leg_a
        meta: dict[str, Any] = scorer.to_meta()
        meta["exit_config"] = {
            "trailing_enabled": False,
            "partial_exits": [(1.5, 0.5)],
            "max_hold_minutes": self.max_hold_days * 390,
            "vol_adaptive": False,
        }
        meta["pair"] = key
        meta["z_score"] = round(z_score, 4)
        meta["hedge_ratio"] = round(hedge_ratio, 6)
        meta["spread"] = round(spread, 6)
        meta["other_leg"] = other_leg
        if corr is not None:
            meta["rolling_correlation"] = round(corr, 4)

        self.last_signal_time[symbol] = bar.time
        ps["signal_active"] = True

        self.logger.info(
            "pair_trading_v2: entry signal",
            symbol=symbol,
            pair=key,
            side=side.value,
            z_score=round(z_score, 4),
            hedge_ratio=round(hedge_ratio, 6),
            confluence=round(scorer.score(), 2),
            correlation=round(corr, 4) if corr is not None else None,
        )

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=scorer.conviction_multiplier(),
            meta=meta,
        )

    # -- external helpers -------------------------------------------------

    def mark_pair_inactive(self, pair_key: str) -> None:
        """Called by execution engine when both legs of a pair trade close."""
        ps = self._pair_state.get(pair_key)
        if ps is not None:
            ps["signal_active"] = False
