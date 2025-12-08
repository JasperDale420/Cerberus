from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from src.core.logger import StructuredLogger
from src.strategies.base import Signal, SymbolState, MarketState
from src.analysis.regime import Regime

@dataclass
class OrderIntent:
    symbol: str
    side: str
    qty: float
    order_type: str # "market", "limit"
    limit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy: str
    correlation_id: Optional[str] = None

class RiskManager:
    """
    Enforces risk limits and converts Signals to OrderIntents.
    """
    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.max_daily_loss = config.get("max_daily_loss", 1000.0)
        self.max_risk_per_trade = config.get("max_risk_per_trade", 50.0) # In dollars
        self.current_daily_pnl = 0.0

    def apply(self, 
              signal: Signal, 
              symbol_state: SymbolState, 
              market_state: MarketState) -> Optional[OrderIntent]:
        """
        Evaluates a signal and returns an OrderIntent if approved, or None if rejected.
        """
        # 1. Check Daily Loss Limit
        if self.current_daily_pnl <= -self.max_daily_loss:
            self.logger.warning("Signal rejected: Max daily loss exceeded", 
                                current_pnl=self.current_daily_pnl, 
                                limit=self.max_daily_loss)
            return None

        # 2. Calculate Position Size based on Risk
        # Risk = |Entry - Stop| * Qty
        # Qty = MaxRisk / |Entry - Stop|
        
        risk_per_share = abs(signal.entry_price - signal.stop_price)
        if risk_per_share <= 0:
            self.logger.warning("Signal rejected: Invalid stop price (zero risk per share)", signal=signal)
            return None
            
        qty = self.max_risk_per_trade / risk_per_share
        qty = round(qty) # Round to nearest share (assuming no fractional for now)
        
        if qty <= 0:
            self.logger.warning("Signal rejected: Calculated quantity is zero", signal=signal)
            return None

        # 3. Create Order Intent
        intent = OrderIntent(
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            order_type="limit", # Default to limit for safety
            limit_price=signal.entry_price,
            stop_loss=signal.stop_price,
            take_profit=signal.target_price,
            strategy=signal.strategy
        )
        
        self.logger.info("Signal approved", intent=intent)
        return intent

    def update_pnl(self, pnl: float):
        """
        Updates the current daily PnL.
        """
        self.current_daily_pnl += pnl
