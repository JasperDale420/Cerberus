from datetime import timedelta
from typing import Any

from src.core.config import ConfigLoader
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.strategies.base import BaseStrategy


class AlpacaOptionsTrend(BaseStrategy):
    name = "alpaca_options_trend"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.trend_threshold = float(config.get("trend_threshold", 1.5))
        self.alpaca_client = AlpacaClient(ConfigLoader(), logger)

    def on_bar(
        self, symbol: str, bar: Bar, symbol_state: SymbolState, market_state: MarketState
    ) -> Signal | None:
        if not self._check_cooldown(symbol, bar.time):
            return None

        # We need an option contract symbol to fetch trades. We can simulate taking the first call option.
        # In a real scenario, we'd query the chain first to find the ATM option.
        try:
            # Fetch chain to get an option symbol
            chain = self.alpaca_client.get_historical_option_chain(symbol)
            if chain:
                first_option_symbol = list(chain.keys())[0]

                # Fetch recent trades for this specific option contract
                end_time = bar.time
                start_time = end_time - timedelta(minutes=5)
                trades = self.alpaca_client.get_historical_option_trades(first_option_symbol, start=start_time, end=end_time)

                self.logger.info("Fetched recent option trades for trend analysis", option_symbol=first_option_symbol, num_trades=len(trades))

                # Simulate trend detection from massive option trades
                if len(trades) > 50:
                    entry = bar.close
                    stop = entry * 0.95
                    target = entry * 1.05

                    self.logger.info("Bullish options trend detected from Trade Tape", symbol=symbol, option_symbol=first_option_symbol)

                    self.last_signal_time[symbol] = bar.time
                    return self._create_signal(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        bar=bar,
                        market_state=market_state,
                        stop_price=stop,
                        target_price=target,
                        size_hint=0.1,
                        meta={"tape_signal": "bull_sweeps"}
                    )
        except Exception as e:
            self.logger.warning("Failed to fetch option trades for trend", symbol=symbol, error=str(e))

        return None
