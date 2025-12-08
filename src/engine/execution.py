from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime
from src.core.logger import StructuredLogger
from src.analysis.regime import RegimeDetector, Regime
from src.strategies.base import BaseStrategy, SymbolState, MarketState, Signal
from src.data.alpaca import AlpacaClient
from src.engine.risk import RiskManager, OrderIntent
from src.engine.orders import OrderExecutor
from src.scanner.core import Scanner

class ExecutionEngine:
    """
    Orchestrates data flow, strategy execution, and order management.
    """
    def __init__(self, 
                 config: Dict[str, Any], 
                 logger: StructuredLogger,
                 alpaca_client: Optional[AlpacaClient] = None):
        self.config = config
        self.logger = logger
        self.alpaca_client = alpaca_client
        
        self.regime_detector = RegimeDetector(logger=logger)
        self.risk_manager = RiskManager(config, logger)
        self.order_executor = OrderExecutor(alpaca_client, logger) if alpaca_client else None
        self.scanner: Optional[Scanner] = None # Injected later or via init
        
        self.strategies: Dict[str, BaseStrategy] = {}
        self.symbol_states: Dict[str, SymbolState] = {}
        self.market_state = MarketState(
            time=datetime.utcnow(),
            regime=Regime.CHOP
        )

    def register_strategy(self, strategy: BaseStrategy):
        """
        Registers a strategy with the engine.
        """
        self.strategies[strategy.name] = strategy
        self.logger.info("Strategy registered", strategy=strategy.name)

    def on_bar(self, symbol: str, bar: Any):
        """
        Process a new bar for a symbol.
        """
        # Update Market State (if symbol is index)
        if symbol == self.config.get("index_symbol", "SPY"):
            self.market_state.regime = self.regime_detector.update(bar.close)
            self.market_state.time = datetime.utcnow() # Should use bar time in real app

        # Update Symbol State
        if symbol not in self.symbol_states:
            self.symbol_states[symbol] = SymbolState(
                symbol=symbol,
                bars=deque(maxlen=100),
                position=None
            )
        
        state = self.symbol_states[symbol]
        state.bars.append(bar)

        # Run Strategies
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.on_bar(symbol, bar, state, self.market_state)
                if signal:
                    self.logger.info("Signal generated", signal=signal)
                    self._process_signal(signal)
            except Exception as e:
                self.logger.error("Strategy execution failed", strategy=name, symbol=symbol, error=str(e))

    def _process_signal(self, signal: Signal):
        """
        Passes signal to Risk Manager and then Order Executor.
        """
        self.logger.info("Processing signal", signal=signal)
        
        # 1. Risk Check
        intent = self.risk_manager.apply(signal, self.symbol_states[signal.symbol], self.market_state)
        
        if not intent:
            self.logger.info("Signal rejected by Risk Manager")
            return

        # 2. Order Execution
        if self.order_executor:
            try:
                self.order_executor.submit(intent)
            except Exception as e:
                self.logger.error("Order execution failed", error=str(e))
        else:
            self.logger.warning("No OrderExecutor available (paper/mock mode)", intent=intent)

    async def run_scan(self):
        """
        Triggers a scan and updates the active symbol list/strategies.
        """
        if not self.scanner:
            self.logger.warning("Scanner not initialized")
            return

        results = await self.scanner.scan()
        
        # Update symbol states or active list
        # For this slice, we just log the results and maybe init states
        for res in results:
            if res.symbol not in self.symbol_states:
                self.symbol_states[res.symbol] = SymbolState(res.symbol, [], None)
            
            self.logger.info("Scanner result", symbol=res.symbol, strategies=res.matching_strategies)

