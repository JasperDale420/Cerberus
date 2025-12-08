from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None

@dataclass
class Trade:
    symbol: str
    timestamp: datetime
    price: float
    size: float
    exchange: Optional[str] = None
    id: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    tape: Optional[str] = None

@dataclass
class Quote:
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    bid_exchange: Optional[str] = None
    ask_exchange: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    tape: Optional[str] = None

@dataclass
class SymbolFeatures:
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    # Add other features as needed (e.g., from Unusual Whales)
    flow_sentiment: Optional[float] = None
    volatility: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
