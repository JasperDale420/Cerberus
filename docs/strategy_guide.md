# Strategy Development Guide

Strategies are implemented in `src/strategies/` and extend `BaseStrategy`.

## Required Interface

Each strategy must implement:

```python
on_bar(
    symbol: str,
    bar: Bar,
    symbol_state: SymbolState,
    market_state: MarketState,
) -> Optional[Signal]
```

Base class helper methods:

- `_require_min_bars(...)`
- `_check_cooldown(...)`
- `_create_signal(...)`
- `update_params(...)`
- `is_past_hard_stop(...)`

Reference: `src/strategies/base.py`.

## Minimal Example

```python
from typing import Any, Dict, Optional
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        if not self._require_min_bars(symbol_state, 20):
            return None
        if not self._check_cooldown(symbol, bar.time):
            return None

        # Example condition
        if bar.close > bar.open:
            return self._create_signal(
                symbol=symbol,
                side=OrderSide.BUY,
                bar=bar,
                market_state=market_state,
                stop_price=bar.close * 0.99,
                target_price=bar.close * 1.02,
                meta={"reason": "demo"},
            )
        return None
```

## Registration

Strategies are registered in `src/main.py` via the `strategy_registry` map.

To add a strategy:

1. Create `src/strategies/<name>.py`
2. Add config under `strategies.<name>` in YAML
3. Import and register in `src/main.py`
4. Add tests under `tests/`

## Config Patterns

Common strategy config keys:

- `enabled`
- `cooldown_bars`
- `bar_duration_minutes`
- strategy-specific thresholds
- optional `hard_stop_time`

## Testing Guidance

- Unit test signal conditions and edge cases.
- Verify no signal when bars/features are insufficient.
- Verify cooldown behavior.
- Add integration/backtest tests for order flow impact.

Useful commands:

```bash
make test-unit
make test-integration
```
