from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Side, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class PairTradingStrategy(BaseStrategy):
    """
    Executes trades for cointegrated pairs based on scanner metadata.
    Fades spread deviations (StatArb) and exits on mean reversion.
    """

    name = "pair_trading"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        # Thresholds usually come from PairTradingConfig but can be overridden here
        self.entry_zscore = float(config.get("entry_zscore", 2.0))
        self.exit_zscore = float(config.get("exit_zscore", 0.0))

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """
        Process bar for pair trading.
        """
        pair_info = symbol_state.meta.get("pair_trade")
        if not pair_info:
            return None

        if symbol_state.position is None:
            return self._handle_entry(
                symbol, bar, symbol_state, market_state, pair_info
            )
        else:
            return self._handle_exit(symbol, bar, symbol_state, market_state, pair_info)

    def _handle_entry(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
        pair_info: Dict[str, Any],
    ) -> Optional[Signal]:
        z_score = pair_info.get("z_score", 0.0)
        side_val = pair_info.get("side")

        if not self._check_cooldown(symbol, bar.time):
            return None

        if abs(z_score) < self.entry_zscore:
            return None

        side = OrderSide.BUY if side_val == "buy" else OrderSide.SELL

        # Safety stops (ATR-based)
        atr = symbol_state.meta.get("atr", bar.close * 0.02)
        stop_dist = atr * 4.0

        stop_price = (
            bar.close - stop_dist if side == OrderSide.BUY else bar.close + stop_dist
        )
        target_price = (
            bar.close + stop_dist * 2
            if side == OrderSide.BUY
            else bar.close - stop_dist * 2
        )

        self.last_signal_time[symbol] = bar.time
        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta={"pair_trade": pair_info},
        )

    def _handle_exit(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
        pair_info: Dict[str, Any],
    ) -> Optional[Signal]:
        z_score = pair_info.get("z_score", 0.0)
        pos = symbol_state.position

        should_exit = False
        if pos.side == Side.LONG and z_score >= self.exit_zscore:
            should_exit = True
            reason = f"Mean reversion: Z-score {z_score:.2f} >= {self.exit_zscore}"
        elif pos.side == Side.SHORT and z_score <= -self.exit_zscore:
            should_exit = True
            reason = f"Mean reversion: Z-score {z_score:.2f} <= -{self.exit_zscore}"

        if should_exit:
            self.logger.info(
                "Pair trading exit signal emitted",
                symbol=symbol,
                reason=reason,
                z_score=z_score,
            )
            return self._create_signal(
                symbol=symbol,
                side=OrderSide.SELL if pos.side == Side.LONG else OrderSide.BUY,
                bar=bar,
                market_state=market_state,
                stop_price=0.0,
                target_price=0.0,
                meta={"exit_reason": "mean_reversion", "z_score": z_score},
            )
        return None
