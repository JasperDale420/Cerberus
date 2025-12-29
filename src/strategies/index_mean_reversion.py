from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy
from src.strategies.config_models import IndexMeanReversionConfig


class IndexMeanReversionStrategy(BaseStrategy):
    """
    Index Mean Reversion.
    Fades deviations (> 2 sigma) in Index ETFs during CHOP regimes.
    """

    name = "index_mean_reversion"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        cfg = IndexMeanReversionConfig(**config)
        self.bb_len = cfg.bb_len
        self.bb_std = cfg.bb_std
        self.risk_reward = cfg.risk_reward
        self.stop_std = cfg.stop_std
        self.stop_pct = cfg.stop_pct
        self.allowed_symbols = set(cfg.symbols)

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # 1. Regime Check (Strictly CHOP)
        # PRD: only for index ETFs (default SPY/QQQ).
        if str(symbol).upper() not in self.allowed_symbols:
            return None
        if market_state.regime != Regime.CHOP:
            return None

        # Require enough history for BB calculation
        if not self._require_min_bars(symbol_state, self.bb_len + 1):
            return None

        bars = list(symbol_state.bars)
        # Prefer cached rolling mean/std from engine; fall back to local computation.
        mean = symbol_state.indicators.get(f"bb_mean:{int(self.bb_len)}")
        std = symbol_state.indicators.get(f"bb_std:{int(self.bb_len)}")
        if mean is None or std is None:
            closes = [float(b.close) for b in bars[-int(self.bb_len) :]]
            if len(closes) < int(self.bb_len):
                return None
            mean = sum(closes) / len(closes)
            m = float(mean)
            var = sum((x - m) ** 2 for x in closes) / max(1, len(closes))
            std = float(var**0.5)
        try:
            mean_f = float(mean)
            std_f = float(std)
        except Exception:
            return None

        current_bbm = mean_f
        current_bbu = mean_f + (std_f * float(self.bb_std))
        current_bbl = mean_f - (std_f * float(self.bb_std))

        # 4. Check Signals
        signal = None
        price = bar.close

        # LONG: Price < Lower Band (Oversold)
        if price < current_bbl:
            # Revert to Mean
            # Stop: Further deviation or fixed %
            stop_price = price * (1.0 - self.stop_pct)
            target_price = current_bbm

            # Sanity check: is target > price?
            if target_price > price:
                signal = self._create_signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    bar=bar,
                    market_state=market_state,
                    stop_price=stop_price,
                    target_price=target_price,
                    meta={"z_score": -2.0, "full_reversion": True},
                )

        # SHORT: Price > Upper Band (Overbought)
        elif price > current_bbu:
            # Revert to Mean
            stop_price = price * (1.0 + self.stop_pct)
            target_price = current_bbm

            if target_price < price:
                signal = self._create_signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    bar=bar,
                    market_state=market_state,
                    stop_price=stop_price,
                    target_price=target_price,
                    meta={"z_score": 2.0, "full_reversion": True},
                )

        return signal
