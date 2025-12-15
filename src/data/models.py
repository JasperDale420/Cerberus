from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


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
