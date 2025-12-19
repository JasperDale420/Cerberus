from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class IndexMeanReversionStrategy(BaseStrategy):
    """
    Index Mean Reversion.
    Fades deviations (> 2 sigma) in Index ETFs during CHOP regimes.
    """

    name = "index_mean_reversion"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.bb_len = config.get("bb_len", 20)
        self.bb_std = config.get("bb_std", 2.0)
        self.risk_reward = config.get("risk_reward", 2.0)
        self.stop_std = config.get(
            "stop_std", 3.0
        )  # Stop at 3 sigma deviation? Or fixed pct?
        # Fixed pct might be safer for index scalping.
        self.stop_pct = config.get(
            "stop_pct", 0.005
        )  # 0.5% stop on index is decent width for M5
        self.allowed_symbols = {
            str(s).upper() for s in (config.get("symbols") or ["SPY", "QQQ"])
        }

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

        # 2. History Check
        if not symbol_state.bars or len(symbol_state.bars) < self.bb_len + 1:
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
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    size_hint=0,
                    entry_price=price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy=self.name,
                    regime=market_state.regime,
                    generated_at=bar.time,
                    meta={"z_score": -2.0, "full_reversion": True},
                )

        # SHORT: Price > Upper Band (Overbought)
        elif price > current_bbu:
            # Revert to Mean
            stop_price = price * (1.0 + self.stop_pct)
            target_price = current_bbm

            if target_price < price:
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    size_hint=0,
                    entry_price=price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy=self.name,
                    regime=market_state.regime,
                    generated_at=bar.time,
                    meta={"z_score": 2.0, "full_reversion": True},
                )

        return signal
