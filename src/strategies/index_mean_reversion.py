from typing import Any, Dict, Optional

import pandas as pd
import pandas_ta as ta

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

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # 1. Regime Check (Strictly CHOP)
        if market_state.regime != Regime.CHOP:
            return None

        # 2. History Check
        if not symbol_state.bars or len(symbol_state.bars) < self.bb_len + 1:
            return None

        bars = list(symbol_state.bars)
        df = pd.DataFrame([vars(b) for b in bars])
        close = df["close"].astype(float)

        # 3. Calculate Bollinger Bands
        bb = ta.bbands(close, length=self.bb_len, std=self.bb_std)
        if bb is None:
            return None

        # Cols: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        # Need to match dynamic names
        lower_col = f"BBL_{self.bb_len}_{float(self.bb_std)}"
        upper_col = f"BBU_{self.bb_len}_{float(self.bb_std)}"
        mean_col = f"BBM_{self.bb_len}_{float(self.bb_std)}"

        if lower_col not in bb.columns:
            return None

        current_bbl = bb[lower_col].iloc[-1]
        current_bbu = bb[upper_col].iloc[-1]
        current_bbm = bb[mean_col].iloc[-1]

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
                    correlation_id=f"{self.name}-long-{symbol}-{bar.time.timestamp()}",
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
                    correlation_id=f"{self.name}-short-{symbol}-{bar.time.timestamp()}",
                )

        return signal
