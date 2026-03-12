from typing import Any

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class CointegrationPairsStrategy(BaseStrategy):
    """
    Continuous Co-integration Pairs Trading Strategy.
    Monitors the spread between two historically cointegrated assets.
    When the Z-score of the spread deviates significantly from the mean,
    it trades the divergence expecting mean-reversion.

    For standalone functionality in this system, it utilizes the
    `cluster_residual` (spread Z-score proxy) initialized in the feature pipeline.
    """

    name: str = "cointegration_pairs"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        self.entry_z_threshold = float(config.get("entry_z_threshold", 2.0))
        self.stop_z_threshold = float(config.get("stop_z_threshold", 3.5))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        if not self._check_cooldown(symbol, bar.time):
            return None

        features = symbol_state.meta.get("features")
        if not features:
            return None

        # Simulate co-integration spread using the cluster_residual
        z_score = getattr(features, "cluster_residual", 0.0)

        if abs(z_score) < self.entry_z_threshold:
            return None

        # Mean-reversion logic:
        # If Z-score is positive and high (outperforming its pair), we short it.
        # If Z-score is negative and low (underperforming its pair), we long it.
        if z_score > self.entry_z_threshold:
            side = OrderSide.SELL
        elif z_score < -self.entry_z_threshold:
            side = OrderSide.BUY
        else:
            return None

        atr = float(symbol_state.meta.get("atr", bar.close * 0.02) or bar.close * 0.02)

        # Stop and Target distances
        stop_dist = max((self.stop_z_threshold - self.entry_z_threshold) * (atr / 2.0), atr)
        target_dist = abs(z_score) * (atr / 2.0)

        if side == OrderSide.BUY:
            stop_price = bar.close - stop_dist
            target_price = bar.close + target_dist
        else:
            stop_price = bar.close + stop_dist
            target_price = bar.close - target_dist

        self.last_signal_time[symbol] = bar.time

        self.logger.info(
            "cointegration_pairs: entry signal",
            symbol=symbol,
            side=side.value,
            z_score=round(z_score, 4),
        )

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=1.0,
            meta={"z_score": z_score, "pair_correlation_assumed": True},
        )
