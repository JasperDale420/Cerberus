# Backtest & WFO Robustness Upgrade — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Cerberus backtest and WFO systems more realistic, analytically rich, and actionable — with pluggable fill models, per-strategy overnight handling, Monte Carlo validation, data quality checks, benchmark comparison, automated holdout, parameter sensitivity, diagnostics, and EmpireUI dashboard enhancements.

**Architecture:** Layered composition on top of the existing working engine. New capabilities live in separate modules with clean interfaces. The backtest runner orchestrates them via config flags. EmpireUI gets three new views served by a lightweight FastAPI service reading JSON result files.

**Tech Stack:** Python 3.12+, pytest 8.3+, Optuna, NumPy/SciPy, Recharts (EmpireUI), TanStack Query, FastAPI (new API layer).

**Design Doc:** `docs/plans/2026-03-19-backtest-wfo-robustness-design.md`

---

## Phase 1: Pluggable Fill Model

### Task 1.1: FillModel Protocol & FillResult

**Files:**
- Create: `src/backtest/fill_models/__init__.py`
- Create: `src/backtest/fill_models/protocol.py`
- Test: `tests/unit/test_fill_models.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_fill_models.py
import pytest
from src.backtest.fill_models.protocol import FillModel, FillResult

@pytest.mark.unit
def test_fill_result_has_required_fields():
    result = FillResult(
        fill_price=150.25,
        filled_qty=100,
        commission=0.10,
        slippage_bps=2.0,
        market_impact=0.05,
    )
    assert result.fill_price == 150.25
    assert result.filled_qty == 100
    assert result.commission == 0.10
    assert result.slippage_bps == 2.0
    assert result.market_impact == 0.05


@pytest.mark.unit
def test_fill_model_is_protocol():
    """FillModel should be a runtime-checkable Protocol."""
    assert hasattr(FillModel, "compute_fill")
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_fill_models.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/backtest/fill_models/__init__.py
from src.backtest.fill_models.protocol import FillModel, FillResult

__all__ = ["FillModel", "FillResult"]
```

```python
# src/backtest/fill_models/protocol.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core.domain import Bar, OrderIntent


@runtime_checkable
class FillModel(Protocol):
    """Protocol for pluggable fill simulation models."""

    def compute_fill(
        self,
        order_side: str,
        order_qty: int,
        order_price: float | None,
        order_type: str,
        bar: Bar,
    ) -> FillResult: ...


@dataclass(frozen=True, slots=True)
class FillResult:
    fill_price: float
    filled_qty: int
    commission: float
    slippage_bps: float
    market_impact: float
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_fill_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/backtest/fill_models/ tests/unit/test_fill_models.py
git commit -m "feat(backtest): add FillModel protocol and FillResult dataclass"
```

---

### Task 1.2: FixedSlippageFillModel (current behavior extracted)

**Files:**
- Create: `src/backtest/fill_models/fixed.py`
- Modify: `tests/unit/test_fill_models.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_fill_models.py (append)
from src.backtest.fill_models.fixed import FixedSlippageFillModel


@pytest.mark.unit
def test_fixed_fill_model_buy_slippage():
    model = FixedSlippageFillModel(slippage_bps=2.0, commission_per_share=0.001)
    result = model.compute_fill(
        order_side="buy",
        order_qty=100,
        order_price=150.00,
        order_type="market",
        bar=None,  # Not used for fixed model
    )
    expected_price = 150.00 * (1 + 2.0 / 10_000)
    assert result.fill_price == pytest.approx(expected_price)
    assert result.filled_qty == 100
    assert result.commission == pytest.approx(0.10)
    assert result.slippage_bps == 2.0
    assert result.market_impact == 0.0


@pytest.mark.unit
def test_fixed_fill_model_sell_slippage():
    model = FixedSlippageFillModel(slippage_bps=2.0, commission_per_share=0.001)
    result = model.compute_fill(
        order_side="sell",
        order_qty=50,
        order_price=200.00,
        order_type="market",
        bar=None,
    )
    expected_price = 200.00 * (1 - 2.0 / 10_000)
    assert result.fill_price == pytest.approx(expected_price)
    assert result.filled_qty == 50
    assert result.commission == pytest.approx(0.05)


@pytest.mark.unit
def test_fixed_fill_model_satisfies_protocol():
    from src.backtest.fill_models.protocol import FillModel

    model = FixedSlippageFillModel(slippage_bps=2.0, commission_per_share=0.001)
    assert isinstance(model, FillModel)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_fill_models.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# src/backtest/fill_models/fixed.py
from __future__ import annotations

from typing import TYPE_CHECKING

from src.backtest.fill_models.protocol import FillResult

if TYPE_CHECKING:
    from src.core.domain import Bar


class FixedSlippageFillModel:
    """Original fixed-BPS slippage model. Extracted from SimulatedOrderExecutor."""

    def __init__(self, slippage_bps: float = 2.0, commission_per_share: float = 0.001):
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share

    def compute_fill(
        self,
        order_side: str,
        order_qty: int,
        order_price: float | None,
        order_type: str,
        bar: Bar | None,
    ) -> FillResult:
        price = order_price or 0.0
        slip_frac = self.slippage_bps / 10_000.0
        if order_side == "buy":
            fill_price = price * (1.0 + slip_frac)
        else:
            fill_price = price * (1.0 - slip_frac)

        commission = self.commission_per_share * order_qty
        return FillResult(
            fill_price=fill_price,
            filled_qty=order_qty,
            commission=commission,
            slippage_bps=self.slippage_bps,
            market_impact=0.0,
        )
```

**Step 4: Run tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_fill_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/backtest/fill_models/fixed.py tests/unit/test_fill_models.py
git commit -m "feat(backtest): extract FixedSlippageFillModel from executor"
```

---

### Task 1.3: VolumeAwareFillModel

**Files:**
- Create: `src/backtest/fill_models/volume_aware.py`
- Modify: `tests/unit/test_fill_models.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_fill_models.py (append)
from unittest.mock import SimpleNamespace
from src.backtest.fill_models.volume_aware import VolumeAwareFillModel


@pytest.mark.unit
def test_volume_aware_low_participation():
    """Small order relative to volume — slippage close to base."""
    model = VolumeAwareFillModel(base_slippage_bps=2.0, impact_coefficient=200.0, commission_per_share=0.001)
    bar = SimpleNamespace(volume=100_000)
    result = model.compute_fill(
        order_side="buy", order_qty=100, order_price=150.00, order_type="market", bar=bar,
    )
    # participation = 100/100_000 = 0.001
    # effective_slip = 2.0 + (0.001 * 200) = 2.2 bps
    expected_price = 150.00 * (1 + 2.2 / 10_000)
    assert result.fill_price == pytest.approx(expected_price, rel=1e-6)
    assert result.slippage_bps == pytest.approx(2.2)
    assert result.market_impact == pytest.approx(0.2)


@pytest.mark.unit
def test_volume_aware_high_participation():
    """Large order relative to volume — significant additional slippage."""
    model = VolumeAwareFillModel(base_slippage_bps=2.0, impact_coefficient=200.0, commission_per_share=0.001)
    bar = SimpleNamespace(volume=1_000)
    result = model.compute_fill(
        order_side="buy", order_qty=100, order_price=150.00, order_type="market", bar=bar,
    )
    # participation = 100/1_000 = 0.10
    # effective_slip = 2.0 + (0.10 * 200) = 22.0 bps
    assert result.slippage_bps == pytest.approx(22.0)
    assert result.market_impact == pytest.approx(20.0)


@pytest.mark.unit
def test_volume_aware_zero_volume_bar_uses_max_slippage():
    """Zero volume bar — cap slippage at max_slippage_bps."""
    model = VolumeAwareFillModel(
        base_slippage_bps=2.0, impact_coefficient=200.0,
        commission_per_share=0.001, max_slippage_bps=50.0,
    )
    bar = SimpleNamespace(volume=0)
    result = model.compute_fill(
        order_side="buy", order_qty=100, order_price=150.00, order_type="market", bar=bar,
    )
    assert result.slippage_bps == pytest.approx(50.0)


@pytest.mark.unit
def test_volume_aware_sell_slippage_direction():
    model = VolumeAwareFillModel(base_slippage_bps=2.0, impact_coefficient=200.0, commission_per_share=0.001)
    bar = SimpleNamespace(volume=100_000)
    result = model.compute_fill(
        order_side="sell", order_qty=100, order_price=150.00, order_type="market", bar=bar,
    )
    assert result.fill_price < 150.00  # Sell gets worse price


@pytest.mark.unit
def test_volume_aware_satisfies_protocol():
    from src.backtest.fill_models.protocol import FillModel

    model = VolumeAwareFillModel()
    assert isinstance(model, FillModel)
```

**Step 2: Run to verify failure**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_fill_models.py -v -k "volume_aware"`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# src/backtest/fill_models/volume_aware.py
from __future__ import annotations

from typing import TYPE_CHECKING

from src.backtest.fill_models.protocol import FillResult

if TYPE_CHECKING:
    from src.core.domain import Bar


class VolumeAwareFillModel:
    """Slippage scales with order participation rate in bar volume."""

    def __init__(
        self,
        base_slippage_bps: float = 2.0,
        impact_coefficient: float = 200.0,
        commission_per_share: float = 0.001,
        max_slippage_bps: float = 50.0,
    ):
        self.base_slippage_bps = base_slippage_bps
        self.impact_coefficient = impact_coefficient
        self.commission_per_share = commission_per_share
        self.max_slippage_bps = max_slippage_bps

    def compute_fill(
        self,
        order_side: str,
        order_qty: int,
        order_price: float | None,
        order_type: str,
        bar: Bar | None,
    ) -> FillResult:
        price = order_price or 0.0
        bar_volume = getattr(bar, "volume", 0) if bar else 0

        if bar_volume > 0:
            participation = order_qty / bar_volume
            impact_bps = participation * self.impact_coefficient
            effective_bps = min(self.base_slippage_bps + impact_bps, self.max_slippage_bps)
        else:
            effective_bps = self.max_slippage_bps
            impact_bps = effective_bps - self.base_slippage_bps

        slip_frac = effective_bps / 10_000.0
        if order_side == "buy":
            fill_price = price * (1.0 + slip_frac)
        else:
            fill_price = price * (1.0 - slip_frac)

        commission = self.commission_per_share * order_qty
        return FillResult(
            fill_price=fill_price,
            filled_qty=order_qty,
            commission=commission,
            slippage_bps=effective_bps,
            market_impact=impact_bps,
        )
```

**Step 4: Run tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_fill_models.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/backtest/fill_models/volume_aware.py tests/unit/test_fill_models.py
git commit -m "feat(backtest): add VolumeAwareFillModel with participation-rate scaling"
```

---

### Task 1.4: Integrate FillModel into SimulatedOrderExecutor

**Files:**
- Modify: `src/backtest/executor.py:20-41` (constructor), `243-257` (slippage/commission), `259-365` (process_bar)
- Modify: `src/backtest/fill_models/__init__.py`
- Create: `tests/unit/test_executor_fill_model.py`

**Step 1: Write failing test**

```python
# tests/unit/test_executor_fill_model.py
import pytest
from unittest.mock import MagicMock, SimpleNamespace
from src.backtest.fill_models.volume_aware import VolumeAwareFillModel
from src.backtest.fill_models.fixed import FixedSlippageFillModel


@pytest.mark.unit
def test_executor_uses_injected_fill_model():
    """SimulatedOrderExecutor should delegate fill calculation to FillModel."""
    from src.backtest.executor import SimulatedOrderExecutor

    mock_model = MagicMock()
    mock_model.compute_fill.return_value = SimpleNamespace(
        fill_price=150.03, filled_qty=100, commission=0.10,
        slippage_bps=2.0, market_impact=0.0,
    )

    executor = SimulatedOrderExecutor(
        logger=MagicMock(),
        db=None,
        risk_cfg={"slippage_bps": 2.0, "commission_per_share": 0.001},
        fill_model=mock_model,
    )
    # Verify fill_model is stored
    assert executor.fill_model is mock_model


@pytest.mark.unit
def test_executor_defaults_to_fixed_model_when_none():
    """When no fill_model provided, fall back to FixedSlippageFillModel."""
    from src.backtest.executor import SimulatedOrderExecutor

    executor = SimulatedOrderExecutor(
        logger=MagicMock(),
        db=None,
        risk_cfg={"slippage_bps": 2.0, "commission_per_share": 0.001},
    )
    assert isinstance(executor.fill_model, FixedSlippageFillModel)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_executor_fill_model.py -v`
Expected: FAIL — TypeError (unexpected keyword argument 'fill_model')

**Step 3: Modify executor.py**

In `SimulatedOrderExecutor.__init__` (~line 20-41):
- Add `fill_model: FillModel | None = None` parameter
- Default: `self.fill_model = fill_model or FixedSlippageFillModel(slippage_bps=self.slippage_bps, commission_per_share=self.commission_per_share)`

In `process_bar` (~lines 259-365):
- Replace inline `_apply_slippage(price, side)` calls with `self.fill_model.compute_fill(side, qty, price, order_type, bar)`
- Use `result.fill_price` instead of the inline slippage math
- Use `result.commission` instead of `_deduct_commission(qty)`
- Store `result.slippage_bps` and `result.market_impact` in the fill event for analytics

Keep `_apply_slippage` and `_deduct_commission` as private methods but mark them deprecated — they stay for any edge cases but `process_bar` no longer calls them.

**Step 4: Run full executor tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_executor_fill_model.py tests/unit/ -k "executor" -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/backtest/executor.py src/backtest/fill_models/__init__.py tests/unit/test_executor_fill_model.py
git commit -m "feat(backtest): integrate FillModel protocol into SimulatedOrderExecutor"
```

---

### Task 1.5: Fill model config wiring in runner.py

**Files:**
- Modify: `src/backtest/runner.py:580-642` (engine initialization section)

**Step 1: Add factory function**

In `src/backtest/fill_models/__init__.py`, add:

```python
def create_fill_model(config: dict) -> FillModel:
    """Factory: build FillModel from backtest config."""
    model_type = config.get("fill_model", "fixed")
    params = config.get("fill_model_params", {})
    risk_cfg = config.get("risk", {})

    if model_type == "volume_aware":
        return VolumeAwareFillModel(
            base_slippage_bps=params.get("base_slippage_bps", risk_cfg.get("slippage_bps", 2.0)),
            impact_coefficient=params.get("impact_coefficient", 200.0),
            commission_per_share=params.get("commission_per_share", risk_cfg.get("commission_per_share", 0.001)),
            max_slippage_bps=params.get("max_slippage_bps", 50.0),
        )
    else:
        return FixedSlippageFillModel(
            slippage_bps=risk_cfg.get("slippage_bps", 2.0),
            commission_per_share=risk_cfg.get("commission_per_share", 0.001),
        )
```

**Step 2: Wire into runner.py**

In runner.py where `SimulatedOrderExecutor` is constructed (~line 600-610):
- Call `fill_model = create_fill_model(config)` before executor creation
- Pass `fill_model=fill_model` to the executor constructor

**Step 3: Write test for factory**

```python
# tests/unit/test_fill_models.py (append)
from src.backtest.fill_models import create_fill_model
from src.backtest.fill_models.volume_aware import VolumeAwareFillModel
from src.backtest.fill_models.fixed import FixedSlippageFillModel


@pytest.mark.unit
def test_create_fill_model_default_is_fixed():
    model = create_fill_model({})
    assert isinstance(model, FixedSlippageFillModel)


@pytest.mark.unit
def test_create_fill_model_volume_aware():
    model = create_fill_model({
        "fill_model": "volume_aware",
        "fill_model_params": {"impact_coefficient": 300.0},
        "risk": {"slippage_bps": 3.0},
    })
    assert isinstance(model, VolumeAwareFillModel)
    assert model.impact_coefficient == 300.0
```

**Step 4: Run tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_fill_models.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/backtest/fill_models/__init__.py src/backtest/runner.py tests/unit/test_fill_models.py
git commit -m "feat(backtest): wire fill model factory into backtest runner config"
```

---

## Phase 2: Per-Strategy Overnight Handling

### Task 2.1: Add overnight fields to BaseStrategy

**Files:**
- Modify: `src/strategies/base.py:20-31` (_set_params method)
- Test: `tests/unit/test_overnight_handling.py`

**Step 1: Write failing test**

```python
# tests/unit/test_overnight_handling.py
import pytest
from unittest.mock import MagicMock


@pytest.mark.unit
def test_base_strategy_overnight_defaults():
    """BaseStrategy should have overnight fields with safe defaults."""
    from src.strategies.base import BaseStrategy

    class StubStrategy(BaseStrategy):
        name = "stub"
        def on_bar(self, symbol, bar, symbol_state, market_state):
            return None

    strat = StubStrategy(config={}, logger=MagicMock())
    assert strat.allow_overnight is False
    assert strat.max_hold_days == 0
    assert strat.overnight_stop_mult == 1.0


@pytest.mark.unit
def test_base_strategy_overnight_from_config():
    from src.strategies.base import BaseStrategy

    class StubStrategy(BaseStrategy):
        name = "stub"
        def on_bar(self, symbol, bar, symbol_state, market_state):
            return None

    cfg = {"allow_overnight": True, "max_hold_days": 5, "overnight_stop_mult": 1.5}
    strat = StubStrategy(config=cfg, logger=MagicMock())
    assert strat.allow_overnight is True
    assert strat.max_hold_days == 5
    assert strat.overnight_stop_mult == 1.5
```

**Step 2: Run to verify failure**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_overnight_handling.py -v`
Expected: FAIL — AttributeError: 'StubStrategy' has no attribute 'allow_overnight'

**Step 3: Add fields to `_set_params` in `base.py`**

In `_set_params` (~lines 20-31), add after existing param setup:

```python
self.allow_overnight = bool(config.get("allow_overnight", False))
self.max_hold_days = int(config.get("max_hold_days", 0))
self.overnight_stop_mult = float(config.get("overnight_stop_mult", 1.0))
```

**Step 4: Run test**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_overnight_handling.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/strategies/base.py tests/unit/test_overnight_handling.py
git commit -m "feat(strategies): add allow_overnight, max_hold_days, overnight_stop_mult to BaseStrategy"
```

---

### Task 2.2: Replace global force_flat with per-strategy EOD logic

**Files:**
- Modify: `src/backtest/runner.py:757-762` (force_flat_at_1600 block)
- Modify: `tests/unit/test_overnight_handling.py`

**Step 1: Write failing test**

```python
# tests/unit/test_overnight_handling.py (append)

@pytest.mark.unit
def test_should_flatten_position_intraday_strategy():
    """Intraday strategies (allow_overnight=False) should flatten at EOD."""
    from src.backtest.runner import _should_flatten_position

    class FakeStrategy:
        allow_overnight = False
        max_hold_days = 0

    assert _should_flatten_position(FakeStrategy(), hold_days=0) is True


@pytest.mark.unit
def test_should_not_flatten_overnight_strategy():
    """Overnight strategies should NOT flatten at EOD."""
    from src.backtest.runner import _should_flatten_position

    class FakeStrategy:
        allow_overnight = True
        max_hold_days = 0

    assert _should_flatten_position(FakeStrategy(), hold_days=0) is False


@pytest.mark.unit
def test_should_flatten_overnight_max_hold_exceeded():
    """Overnight strategies should flatten when max_hold_days exceeded."""
    from src.backtest.runner import _should_flatten_position

    class FakeStrategy:
        allow_overnight = True
        max_hold_days = 3

    assert _should_flatten_position(FakeStrategy(), hold_days=2) is False
    assert _should_flatten_position(FakeStrategy(), hold_days=3) is True
    assert _should_flatten_position(FakeStrategy(), hold_days=5) is True
```

**Step 2: Run to verify failure**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_overnight_handling.py -v -k "should_flatten"`
Expected: FAIL — ImportError (_should_flatten_position)

**Step 3: Implement in runner.py**

Add helper function (near top of runner.py, after imports):

```python
def _should_flatten_position(strategy, hold_days: int) -> bool:
    """Determine if a position should be flattened at EOD."""
    if not strategy.allow_overnight:
        return True
    if strategy.max_hold_days > 0 and hold_days >= strategy.max_hold_days:
        return True
    return False
```

Replace the `force_flat_at_1600` block (~lines 757-762) with per-position logic:

```python
# Per-strategy EOD flatten (replaces global force_flat_at_1600)
et_time = ts.astimezone(_ET_TZ)
if et_time.hour == 15 and et_time.minute >= 55:
    for sym, qty in list(engine.account.positions_qty.items()):
        if abs(qty) < 1e-6:
            continue
        position = engine.position_manager.get_position(sym)
        if position is None:
            continue
        strategy = position.strategy_ref  # Need to store strategy ref on Position
        hold_days = _calc_hold_days(position.entry_time, ts)
        if _should_flatten_position(strategy, hold_days):
            _backtest_flatten_symbol(engine, executor, sym, latest_prices, ts, reason="EOD flatten")
        elif strategy.overnight_stop_mult > 1.0:
            _widen_overnight_stop(executor, sym, strategy.overnight_stop_mult)
```

Note: This requires:
- `Position` to store `strategy_ref` (the strategy object or its overnight config)
- `_backtest_flatten_symbol()` — extract from existing `_backtest_flatten_all()` to flatten a single symbol
- `_widen_overnight_stop()` — find the SL order for this symbol and widen it
- `_calc_hold_days()` — trading days between two timestamps

Implement these helpers. `_calc_hold_days` can use a simple `(current_date - entry_date).days` approximation (close enough for backtests — calendar days / 1.4 ≈ trading days, or count weekdays).

**Step 4: Run tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_overnight_handling.py -v`
Expected: ALL PASS

**Step 5: Run existing backtest tests to verify no regression**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/ -v -k "backtest" --timeout=60`
Expected: ALL PASS

**Step 6: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/backtest/runner.py tests/unit/test_overnight_handling.py
git commit -m "feat(backtest): replace global force_flat with per-strategy overnight handling"
```

---

### Task 2.3: Store strategy reference on Position for overnight lookup

**Files:**
- Modify: `src/core/domain.py` (Position dataclass — add strategy_name field)
- Modify: `src/engine/position_manager.py` (pass strategy name on open)
- Modify: `tests/unit/test_overnight_handling.py`

**Step 1:** Add `strategy_name: str = ""` to Position dataclass in `domain.py`.

**Step 2:** In `PositionManager._open_new_position()`, accept and store `strategy_name` from the fill event's correlation_id or strategy metadata.

**Step 3:** In runner.py, look up strategy by `position.strategy_name` from the strategy registry to get overnight config.

**Step 4:** Test that positions created during backtest carry strategy_name.

**Step 5: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/core/domain.py src/engine/position_manager.py src/backtest/runner.py tests/unit/test_overnight_handling.py
git commit -m "feat(positions): store strategy_name on Position for overnight policy lookup"
```

---

## Phase 3: Data Quality Checks

### Task 3.1: DataQualityReport and checker

**Files:**
- Create: `src/backtest/data_quality.py`
- Test: `tests/unit/test_data_quality.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_data_quality.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone


@pytest.mark.unit
def test_detect_gaps_in_bars():
    from src.backtest.data_quality import check_data_quality

    # Create bars with a 5-bar gap in the middle
    timestamps = pd.date_range("2024-01-02 09:30", periods=100, freq="1min", tz="US/Eastern")
    # Remove bars 50-54 to create gap
    timestamps = timestamps.delete(range(50, 55))
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": 150.0, "high": 151.0, "low": 149.0, "close": 150.5,
        "volume": 1000, "symbol": "AAPL",
    })
    report = check_data_quality({"AAPL": df})
    assert report.symbols["AAPL"].gap_count >= 1
    assert report.symbols["AAPL"].coverage_pct < 100.0


@pytest.mark.unit
def test_detect_zero_volume():
    from src.backtest.data_quality import check_data_quality

    timestamps = pd.date_range("2024-01-02 09:30", periods=50, freq="1min", tz="US/Eastern")
    volumes = [1000] * 50
    volumes[10] = 0
    volumes[20] = 0
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": 150.0, "high": 151.0, "low": 149.0, "close": 150.5,
        "volume": volumes, "symbol": "AAPL",
    })
    report = check_data_quality({"AAPL": df})
    assert report.symbols["AAPL"].zero_volume_bars == 2


@pytest.mark.unit
def test_detect_price_outliers():
    from src.backtest.data_quality import check_data_quality

    timestamps = pd.date_range("2024-01-02 09:30", periods=50, freq="1min", tz="US/Eastern")
    closes = [150.0] * 50
    closes[25] = 200.0  # 33% jump — outlier
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": 150.0, "high": 151.0, "low": 149.0, "close": closes,
        "volume": 1000, "symbol": "AAPL",
    })
    report = check_data_quality({"AAPL": df})
    assert report.symbols["AAPL"].outlier_count >= 1


@pytest.mark.unit
def test_exclude_low_coverage_symbol():
    from src.backtest.data_quality import check_data_quality

    # Only 10 bars when 390 expected for a full day
    timestamps = pd.date_range("2024-01-02 09:30", periods=10, freq="1min", tz="US/Eastern")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": 150.0, "high": 151.0, "low": 149.0, "close": 150.5,
        "volume": 1000, "symbol": "BAD",
    })
    report = check_data_quality({"BAD": df}, min_coverage_pct=80.0, exclude_below_pct=50.0)
    assert "BAD" in report.excluded_symbols


@pytest.mark.unit
def test_detect_stale_prices():
    from src.backtest.data_quality import check_data_quality

    timestamps = pd.date_range("2024-01-02 09:30", periods=50, freq="1min", tz="US/Eastern")
    closes = [150.0] * 50  # All identical — stale
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": 150.0, "high": 151.0, "low": 149.0, "close": closes,
        "volume": 1000, "symbol": "STALE",
    })
    report = check_data_quality({"STALE": df})
    assert report.symbols["STALE"].stale_streak > 10
```

**Step 2: Run to verify failure**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_data_quality.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement data_quality.py**

```python
# src/backtest/data_quality.py
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class SymbolQuality:
    symbol: str
    total_bars: int
    expected_bars: int
    coverage_pct: float
    gap_count: int
    zero_volume_bars: int
    outlier_count: int
    stale_streak: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    symbols: dict[str, SymbolQuality] = field(default_factory=dict)
    excluded_symbols: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_data_quality(
    bars_by_symbol: dict[str, pd.DataFrame],
    min_coverage_pct: float = 80.0,
    exclude_below_pct: float = 50.0,
    max_gap_bars: int = 5,
    outlier_threshold: float = 0.15,
) -> DataQualityReport:
    report = DataQualityReport()

    for symbol, df in bars_by_symbol.items():
        if df.empty:
            report.excluded_symbols.append(symbol)
            continue

        total_bars = len(df)
        trading_days = df["timestamp"].dt.date.nunique()
        expected_bars = max(trading_days * 390, 1)  # 390 1-min bars per day
        coverage_pct = (total_bars / expected_bars) * 100.0

        # Gap detection
        ts_sorted = df["timestamp"].sort_values()
        diffs = ts_sorted.diff().dt.total_seconds().dropna()
        gap_count = int((diffs > max_gap_bars * 60).sum())

        # Zero volume
        zero_volume_bars = int((df["volume"] == 0).sum())

        # Price outliers
        closes = df["close"].values
        returns = np.diff(closes) / closes[:-1]
        outlier_count = int(np.sum(np.abs(returns) > outlier_threshold))

        # Stale prices (max consecutive identical closes)
        stale_streak = 0
        current_streak = 1
        for i in range(1, len(closes)):
            if closes[i] == closes[i - 1]:
                current_streak += 1
                stale_streak = max(stale_streak, current_streak)
            else:
                current_streak = 1

        sq = SymbolQuality(
            symbol=symbol,
            total_bars=total_bars,
            expected_bars=expected_bars,
            coverage_pct=min(coverage_pct, 100.0),
            gap_count=gap_count,
            zero_volume_bars=zero_volume_bars,
            outlier_count=outlier_count,
            stale_streak=stale_streak,
        )

        if coverage_pct < exclude_below_pct:
            report.excluded_symbols.append(symbol)
            sq.warnings.append(f"Excluded: coverage {coverage_pct:.1f}% < {exclude_below_pct}%")
        elif coverage_pct < min_coverage_pct:
            sq.warnings.append(f"Low coverage: {coverage_pct:.1f}%")

        report.symbols[symbol] = sq

    return report
```

**Step 4: Run tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_data_quality.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/backtest/data_quality.py tests/unit/test_data_quality.py
git commit -m "feat(backtest): add pre-backtest data quality checker"
```

---

### Task 3.2: Wire data quality into runner.py

**Files:**
- Modify: `src/backtest/runner.py` (after bars are loaded, before replay loop)

**Step 1:** After bars are loaded (~line 550), call `check_data_quality(bars_by_symbol)`.

**Step 2:** Log warnings for flagged symbols. Remove excluded symbols from the bars dict.

**Step 3:** Run existing backtest tests to verify no regression.

**Step 4: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/backtest/runner.py
git commit -m "feat(backtest): run data quality checks before bar replay"
```

---

## Phase 4: Benchmark Comparison

### Task 4.1: BenchmarkComparison dataclass and computation

**Files:**
- Create: `src/analytics/benchmark.py`
- Test: `tests/unit/test_benchmark.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_benchmark.py
import pytest
import numpy as np


@pytest.mark.unit
def test_benchmark_alpha_positive():
    from src.analytics.benchmark import compute_benchmark_comparison

    # Strategy returns 20%, benchmark returns 10%
    strategy_daily = np.array([0.001] * 200)  # ~20% annual
    benchmark_daily = np.array([0.0005] * 200)  # ~10% annual
    result = compute_benchmark_comparison(strategy_daily, benchmark_daily, "SPY")
    assert result.strategy_alpha > 0
    assert result.benchmark_symbol == "SPY"


@pytest.mark.unit
def test_benchmark_beta_near_zero_for_uncorrelated():
    from src.analytics.benchmark import compute_benchmark_comparison

    rng = np.random.default_rng(42)
    strategy_daily = rng.normal(0.001, 0.01, 200)
    benchmark_daily = rng.normal(0.0005, 0.01, 200)
    result = compute_benchmark_comparison(strategy_daily, benchmark_daily, "SPY")
    # Uncorrelated returns should have beta near 0
    assert abs(result.strategy_beta) < 1.0


@pytest.mark.unit
def test_capture_ratios():
    from src.analytics.benchmark import compute_benchmark_comparison

    # Strategy that captures all up days and none of down days = ideal
    benchmark_daily = np.array([0.01, -0.01, 0.02, -0.005, 0.015])
    strategy_daily = np.array([0.01, 0.0, 0.02, 0.0, 0.015])  # No losses
    result = compute_benchmark_comparison(strategy_daily, benchmark_daily, "SPY")
    assert result.up_capture > 0.9
    assert result.down_capture < 0.1
```

**Step 2: Run to verify failure**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_benchmark.py -v`

**Step 3: Implement**

```python
# src/analytics/benchmark.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    benchmark_symbol: str
    benchmark_return_pct: float
    strategy_return_pct: float
    strategy_alpha: float
    strategy_beta: float
    information_ratio: float
    up_capture: float
    down_capture: float


def compute_benchmark_comparison(
    strategy_daily_returns: np.ndarray,
    benchmark_daily_returns: np.ndarray,
    benchmark_symbol: str,
) -> BenchmarkComparison:
    strat_total = float(np.prod(1 + strategy_daily_returns) - 1)
    bench_total = float(np.prod(1 + benchmark_daily_returns) - 1)

    # Beta via OLS regression
    if np.std(benchmark_daily_returns) > 1e-10:
        cov = np.cov(strategy_daily_returns, benchmark_daily_returns)
        beta = float(cov[0, 1] / cov[1, 1])
    else:
        beta = 0.0

    # Alpha = strategy return - beta * benchmark return
    alpha = strat_total - beta * bench_total

    # Information ratio = alpha / tracking_error
    tracking_error = float(np.std(strategy_daily_returns - benchmark_daily_returns)) * np.sqrt(252)
    ir = alpha / tracking_error if tracking_error > 1e-10 else 0.0

    # Capture ratios
    up_mask = benchmark_daily_returns > 0
    down_mask = benchmark_daily_returns < 0

    if up_mask.sum() > 0:
        up_capture = float(strategy_daily_returns[up_mask].mean() / benchmark_daily_returns[up_mask].mean())
    else:
        up_capture = 0.0

    if down_mask.sum() > 0:
        down_capture = float(strategy_daily_returns[down_mask].mean() / benchmark_daily_returns[down_mask].mean())
    else:
        down_capture = 0.0

    return BenchmarkComparison(
        benchmark_symbol=benchmark_symbol,
        benchmark_return_pct=bench_total * 100,
        strategy_return_pct=strat_total * 100,
        strategy_alpha=alpha * 100,
        strategy_beta=beta,
        information_ratio=ir,
        up_capture=up_capture,
        down_capture=down_capture,
    )
```

**Step 4: Run tests, commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/analytics/benchmark.py tests/unit/test_benchmark.py
git commit -m "feat(analytics): add benchmark comparison with alpha, beta, capture ratios"
```

---

### Task 4.2: Wire benchmark into BacktestReportCard

**Files:**
- Modify: `src/backtest/backtest_report.py` (add benchmark field, compute in `_compute_all`)
- Modify: `src/backtest/runner.py` (pass benchmark daily returns to report)

**Step 1:** Add `benchmark: BenchmarkComparison | None = None` to `ReportMetrics`.

**Step 2:** In runner.py, compute benchmark daily returns from SPY equity curve. Pass to `BacktestReportCard`.

**Step 3:** In `BacktestReportCard._compute_all()`, call `compute_benchmark_comparison()` if benchmark data provided.

**Step 4:** Add benchmark to `write_markdown()` and `to_dict()` outputs.

**Step 5: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/backtest/backtest_report.py src/backtest/runner.py
git commit -m "feat(backtest): integrate benchmark comparison into report card"
```

---

## Phase 5: Monte Carlo Validation

### Task 5.1: MonteCarloResult and bootstrap engine

**Files:**
- Create: `src/analytics/monte_carlo.py`
- Test: `tests/unit/test_monte_carlo.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_monte_carlo.py
import pytest
import numpy as np


@pytest.mark.unit
def test_monte_carlo_basic_properties():
    from src.analytics.monte_carlo import run_monte_carlo

    # 100 trades, mostly profitable
    rng = np.random.default_rng(42)
    trade_pnls = rng.normal(50, 200, 100).tolist()  # avg $50 profit
    result = run_monte_carlo(trade_pnls, initial_capital=100_000, n_simulations=1000)

    assert result.n_simulations == 1000
    assert "sharpe" in result.metric_distributions
    assert "max_drawdown_pct" in result.metric_distributions
    assert "final_equity" in result.metric_distributions
    assert 0.0 <= result.probability_of_loss <= 1.0
    assert 0.0 <= result.probability_of_ruin <= 1.0


@pytest.mark.unit
def test_monte_carlo_all_winners_low_loss_probability():
    from src.analytics.monte_carlo import run_monte_carlo

    trade_pnls = [100.0] * 50  # All winners
    result = run_monte_carlo(trade_pnls, initial_capital=100_000, n_simulations=1000)
    assert result.probability_of_loss == 0.0


@pytest.mark.unit
def test_monte_carlo_all_losers_high_loss_probability():
    from src.analytics.monte_carlo import run_monte_carlo

    trade_pnls = [-100.0] * 50  # All losers
    result = run_monte_carlo(trade_pnls, initial_capital=100_000, n_simulations=1000)
    assert result.probability_of_loss == 1.0


@pytest.mark.unit
def test_monte_carlo_confidence_interval():
    from src.analytics.monte_carlo import run_monte_carlo

    rng = np.random.default_rng(42)
    trade_pnls = rng.normal(50, 200, 200).tolist()
    result = run_monte_carlo(trade_pnls, initial_capital=100_000, n_simulations=5000)
    low, high = result.confidence_interval_95
    assert low < high
    assert low < 110_000  # Shouldn't be absurdly high on the low end


@pytest.mark.unit
def test_monte_carlo_percentile_bands():
    from src.analytics.monte_carlo import run_monte_carlo, PercentileBands

    trade_pnls = [100.0, -50.0, 200.0, -30.0, 150.0] * 20
    result = run_monte_carlo(trade_pnls, initial_capital=100_000, n_simulations=1000)
    bands = result.metric_distributions["final_equity"]
    assert isinstance(bands, PercentileBands)
    assert bands.p5 <= bands.p25 <= bands.p50 <= bands.p75 <= bands.p95
```

**Step 2: Run to verify failure**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/unit/test_monte_carlo.py -v`

**Step 3: Implement**

```python
# src/analytics/monte_carlo.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class PercentileBands:
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    n_simulations: int
    metric_distributions: dict[str, PercentileBands]
    probability_of_loss: float
    probability_of_ruin: float
    worst_case_drawdown: float
    confidence_interval_95: tuple[float, float]


def _compute_percentiles(values: np.ndarray) -> PercentileBands:
    return PercentileBands(
        p5=float(np.percentile(values, 5)),
        p25=float(np.percentile(values, 25)),
        p50=float(np.percentile(values, 50)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
    )


def run_monte_carlo(
    trade_pnls: list[float],
    initial_capital: float = 100_000.0,
    n_simulations: int = 10_000,
    ruin_threshold_pct: float = 30.0,
    seed: int = 42,
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    pnls = np.array(trade_pnls)
    n_trades = len(pnls)

    final_equities = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    sharpes = np.empty(n_simulations)

    for i in range(n_simulations):
        sampled = rng.choice(pnls, size=n_trades, replace=True)
        equity_curve = initial_capital + np.cumsum(sampled)
        final_equities[i] = equity_curve[-1]

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / peak
        max_drawdowns[i] = float(np.max(dd)) * 100.0

        # Sharpe (annualized, assuming ~252 trades/year scaling)
        daily_ish = sampled / initial_capital
        std = np.std(daily_ish)
        sharpes[i] = float(np.mean(daily_ish) / std * np.sqrt(252)) if std > 1e-10 else 0.0

    prob_loss = float(np.mean(final_equities < initial_capital))
    prob_ruin = float(np.mean(max_drawdowns > ruin_threshold_pct))

    return MonteCarloResult(
        n_simulations=n_simulations,
        metric_distributions={
            "final_equity": _compute_percentiles(final_equities),
            "max_drawdown_pct": _compute_percentiles(max_drawdowns),
            "sharpe": _compute_percentiles(sharpes),
        },
        probability_of_loss=prob_loss,
        probability_of_ruin=prob_ruin,
        worst_case_drawdown=float(np.percentile(max_drawdowns, 95)),
        confidence_interval_95=(
            float(np.percentile(final_equities, 2.5)),
            float(np.percentile(final_equities, 97.5)),
        ),
    )
```

**Step 4: Run tests, commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/analytics/monte_carlo.py tests/unit/test_monte_carlo.py
git commit -m "feat(analytics): add Monte Carlo bootstrap simulation engine"
```

---

### Task 5.2: Wire Monte Carlo into backtest report and runner

**Files:**
- Modify: `src/backtest/backtest_report.py` (add monte_carlo field)
- Modify: `src/backtest/runner.py` (call run_monte_carlo after trade records built)

**Step 1:** Add `monte_carlo: MonteCarloResult | None = None` to `ReportMetrics`.

**Step 2:** In runner.py end-of-backtest flow (~line 822), after `_build_trade_records()`:

```python
if config.get("analytics", {}).get("monte_carlo", {}).get("enabled", False):
    mc_cfg = config["analytics"]["monte_carlo"]
    trade_pnls = [t.pnl for t in trade_records]
    mc_result = run_monte_carlo(
        trade_pnls,
        initial_capital=config.get("initial_cash", 100_000),
        n_simulations=mc_cfg.get("n_simulations", 10_000),
        ruin_threshold_pct=mc_cfg.get("ruin_threshold_pct", 30.0),
    )
    report.metrics.monte_carlo = mc_result
```

**Step 3:** Add Monte Carlo to `write_markdown()` and `to_dict()`.

**Step 4: Commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/backtest/backtest_report.py src/backtest/runner.py
git commit -m "feat(backtest): integrate Monte Carlo into backtest report pipeline"
```

---

## Phase 6: WFO Enhancements

### Task 6.1: Automated holdout validation

**Files:**
- Modify: `src/analytics/optuna_harness.py` (~lines 1000-1014, after window loop)
- Test: `tests/unit/test_wfo_holdout.py`

**Step 1: Write failing test**

```python
# tests/unit/test_wfo_holdout.py
import pytest


@pytest.mark.unit
def test_holdout_result_structure():
    from src.analytics.optuna_harness import HoldoutResult

    result = HoldoutResult(
        params_used={"stop_atr_mult": 2.0},
        holdout_sharpe=1.5,
        holdout_pf=2.0,
        holdout_max_dd=8.5,
        holdout_n_trades=25,
        holdout_score=0.75,
        oos_to_holdout_ratio=0.85,
        passed=True,
    )
    assert result.passed is True
    assert result.oos_to_holdout_ratio == 0.85


@pytest.mark.unit
def test_holdout_fails_when_ratio_below_threshold():
    from src.analytics.optuna_harness import HoldoutResult

    result = HoldoutResult(
        params_used={},
        holdout_sharpe=0.3,
        holdout_pf=0.8,
        holdout_max_dd=25.0,
        holdout_n_trades=10,
        holdout_score=0.2,
        oos_to_holdout_ratio=0.3,
        passed=False,
    )
    assert result.passed is False
```

**Step 2: Run to verify failure (HoldoutResult not defined yet)**

**Step 3: Add `HoldoutResult` dataclass and automated holdout logic to `optuna_harness.py`**

After the walk-forward window loop completes (~line 970), add:

```python
# Automated holdout validation
holdout_result = None
if holdout_window and config.get("analytics", {}).get("holdout", {}).get("auto_validate", True):
    # Use params from window with best OOS score
    best_oos_idx = int(np.argmax([s for s in all_oos_scores if s > -100] or [0]))
    holdout_params = best_params_per_window[best_oos_idx]
    holdout_config = _apply_params_to_config(base_config, strategy_name, holdout_params)

    holdout_metrics = run_backtest_for_optimization(
        holdout_window["test_start"], holdout_window["test_end"],
        holdout_config, data_dir,
    )
    holdout_score = composite_objective(holdout_metrics, ...)
    avg_oos = sum(valid_oos) / len(valid_oos) if valid_oos else 0.0
    ratio = holdout_score / avg_oos if avg_oos > 0 else 0.0
    threshold = config.get("analytics", {}).get("holdout", {}).get("pass_threshold", 0.4)

    holdout_result = HoldoutResult(
        params_used=holdout_params,
        holdout_sharpe=holdout_metrics.get("sharpe_ratio", 0),
        holdout_pf=holdout_metrics.get("profit_factor", 0),
        holdout_max_dd=holdout_metrics.get("max_drawdown_pct", 0),
        holdout_n_trades=holdout_metrics.get("n_trades", 0),
        holdout_score=holdout_score,
        oos_to_holdout_ratio=ratio,
        passed=ratio >= threshold,
    )
```

**Step 4: Run tests, commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/analytics/optuna_harness.py tests/unit/test_wfo_holdout.py
git commit -m "feat(wfo): add automated holdout validation after walk-forward windows"
```

---

### Task 6.2: Parameter sensitivity analysis

**Files:**
- Create: `src/analytics/param_sensitivity.py`
- Test: `tests/unit/test_param_sensitivity.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_param_sensitivity.py
import pytest
import numpy as np


@pytest.mark.unit
def test_sensitivity_ranking():
    from src.analytics.param_sensitivity import analyze_param_sensitivity

    # Param A strongly correlated with score, Param B random
    trials_data = {
        "param_a": np.linspace(1.0, 10.0, 50).tolist(),
        "param_b": np.random.default_rng(42).uniform(0, 1, 50).tolist(),
        "score": np.linspace(0.5, 5.0, 50).tolist(),  # Perfectly correlated with param_a
    }
    results = analyze_param_sensitivity(trials_data)
    # param_a should rank higher (more influential)
    a_result = next(r for r in results if r.param_name == "param_a")
    b_result = next(r for r in results if r.param_name == "param_b")
    assert a_result.sensitivity_rank < b_result.sensitivity_rank
    assert abs(a_result.correlation) > abs(b_result.correlation)


@pytest.mark.unit
def test_sensitivity_with_few_trials():
    from src.analytics.param_sensitivity import analyze_param_sensitivity

    trials_data = {
        "param_a": [1.0, 2.0, 3.0],
        "score": [0.5, 1.0, 1.5],
    }
    results = analyze_param_sensitivity(trials_data)
    assert len(results) == 1
    assert results[0].param_name == "param_a"
```

**Step 2: Implement**

```python
# src/analytics/param_sensitivity.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class SensitivityResult:
    param_name: str
    values: list[float]
    scores: list[float]
    correlation: float
    sensitivity_rank: int


def analyze_param_sensitivity(
    trials_data: dict[str, list[float]],
) -> list[SensitivityResult]:
    scores = np.array(trials_data["score"])
    param_names = [k for k in trials_data if k != "score"]

    correlations: list[tuple[str, float, list[float]]] = []
    for name in param_names:
        values = np.array(trials_data[name])
        if np.std(values) < 1e-10:
            correlations.append((name, 0.0, trials_data[name]))
            continue
        corr, _ = stats.spearmanr(values, scores)
        correlations.append((name, abs(float(corr)), trials_data[name]))

    correlations.sort(key=lambda x: x[1], reverse=True)

    results = []
    for rank, (name, abs_corr, values) in enumerate(correlations, start=1):
        signed_corr, _ = stats.spearmanr(np.array(values), scores)
        results.append(SensitivityResult(
            param_name=name,
            values=values,
            scores=trials_data["score"],
            correlation=float(signed_corr),
            sensitivity_rank=rank,
        ))
    return results
```

**Step 3: Run tests, commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/analytics/param_sensitivity.py tests/unit/test_param_sensitivity.py
git commit -m "feat(analytics): add parameter sensitivity analysis with Spearman ranking"
```

---

### Task 6.3: Wire sensitivity into WFO results

**Files:**
- Modify: `src/analytics/optuna_harness.py` (extract trial data from study, call analyze)

After walk-forward completes, extract all trial params + scores from the Optuna study and call `analyze_param_sensitivity()`. Add results to the returned dict.

**Commit:**

```bash
git commit -m "feat(wfo): integrate parameter sensitivity into WFO result output"
```

---

## Phase 7: Post-Backtest Diagnostics Engine

### Task 7.1: DiagnosticsReport and core analyses

**Files:**
- Create: `src/analytics/diagnostics.py`
- Test: `tests/unit/test_diagnostics.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_diagnostics.py
import pytest
from dataclasses import dataclass


@pytest.mark.unit
def test_strategy_ranking_by_pnl():
    from src.analytics.diagnostics import run_diagnostics

    trades = [
        {"strategy": "orb", "pnl": 500, "regime_trend": "UP", "entry_hour": 10, "hold_minutes": 30, "exit_type": "target"},
        {"strategy": "orb", "pnl": -200, "regime_trend": "UP", "entry_hour": 10, "hold_minutes": 45, "exit_type": "stop"},
        {"strategy": "orb", "pnl": 300, "regime_trend": "FLAT", "entry_hour": 11, "hold_minutes": 20, "exit_type": "target"},
        {"strategy": "mean_rev", "pnl": -100, "regime_trend": "UP", "entry_hour": 14, "hold_minutes": 15, "exit_type": "stop"},
        {"strategy": "mean_rev", "pnl": -150, "regime_trend": "DOWN", "entry_hour": 14, "hold_minutes": 25, "exit_type": "time_limit"},
    ]
    report = run_diagnostics(trades, min_trades=2)
    assert report.strategy_rankings[0].strategy == "orb"  # Higher net PnL
    assert report.strategy_rankings[1].strategy == "mean_rev"


@pytest.mark.unit
def test_regime_mismatch_detection():
    from src.analytics.diagnostics import run_diagnostics

    trades = [
        {"strategy": "mean_rev", "pnl": -200, "regime_trend": "UP", "entry_hour": 10, "hold_minutes": 20, "exit_type": "stop"},
        {"strategy": "mean_rev", "pnl": -300, "regime_trend": "UP", "entry_hour": 11, "hold_minutes": 25, "exit_type": "stop"},
        {"strategy": "mean_rev", "pnl": -150, "regime_trend": "UP", "entry_hour": 12, "hold_minutes": 30, "exit_type": "stop"},
        {"strategy": "mean_rev", "pnl": 400, "regime_trend": "FLAT", "entry_hour": 10, "hold_minutes": 15, "exit_type": "target"},
        {"strategy": "mean_rev", "pnl": 300, "regime_trend": "FLAT", "entry_hour": 11, "hold_minutes": 20, "exit_type": "target"},
    ]
    report = run_diagnostics(trades, min_trades=2)
    # Should flag mean_rev losing in UP regime
    mismatches = [m for m in report.regime_mismatches if m.strategy == "mean_rev" and m.regime_value == "UP"]
    assert len(mismatches) == 1
    assert mismatches[0].avg_pnl < 0


@pytest.mark.unit
def test_time_edge_analysis():
    from src.analytics.diagnostics import run_diagnostics

    trades = [
        {"strategy": "orb", "pnl": 500, "regime_trend": "UP", "entry_hour": 10, "hold_minutes": 30, "exit_type": "target"},
        {"strategy": "orb", "pnl": 400, "regime_trend": "UP", "entry_hour": 10, "hold_minutes": 25, "exit_type": "target"},
        {"strategy": "orb", "pnl": -300, "regime_trend": "FLAT", "entry_hour": 14, "hold_minutes": 40, "exit_type": "stop"},
        {"strategy": "orb", "pnl": -250, "regime_trend": "FLAT", "entry_hour": 14, "hold_minutes": 35, "exit_type": "stop"},
    ]
    report = run_diagnostics(trades, min_trades=2)
    time_map = report.time_edge_map["orb"]
    # 10:00 bucket should be profitable, 14:00 bucket should be losing
    morning = next(t for t in time_map if t.hour == 10)
    afternoon = next(t for t in time_map if t.hour == 14)
    assert morning.avg_pnl > 0
    assert afternoon.avg_pnl < 0


@pytest.mark.unit
def test_diagnostics_summary_not_empty():
    from src.analytics.diagnostics import run_diagnostics

    trades = [
        {"strategy": "orb", "pnl": 100, "regime_trend": "UP", "entry_hour": 10, "hold_minutes": 30, "exit_type": "target"},
    ] * 25
    report = run_diagnostics(trades, min_trades=5)
    assert len(report.summary) > 0
```

**Step 2: Implement diagnostics.py**

Build the `run_diagnostics()` function that:
1. Groups trades by strategy → ranks by net PnL
2. Groups by (strategy, regime_trend) → flags where avg_pnl < 0
3. Groups by (strategy, entry_hour) → builds time edge map
4. Groups by (strategy, exit_type) → hold analysis
5. Generates plain-text summary of top findings

Use these dataclasses:

```python
@dataclass
class StrategyRanking:
    strategy: str
    net_pnl: float
    n_trades: int
    sharpe: float

@dataclass
class RegimeMismatch:
    strategy: str
    regime_axis: str
    regime_value: str
    avg_pnl: float
    n_trades: int
    recommendation: str

@dataclass
class TimeSlotEdge:
    hour: int
    avg_pnl: float
    n_trades: int
    win_rate: float

@dataclass
class HoldAnalysis:
    exit_type: str
    avg_pnl: float
    n_trades: int

@dataclass
class DiagnosticsReport:
    strategy_rankings: list[StrategyRanking]
    regime_mismatches: list[RegimeMismatch]
    time_edge_map: dict[str, list[TimeSlotEdge]]
    hold_analysis: dict[str, list[HoldAnalysis]]
    summary: str
```

**Step 3: Run tests, commit**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus
git add src/analytics/diagnostics.py tests/unit/test_diagnostics.py
git commit -m "feat(analytics): add post-backtest diagnostics engine"
```

---

### Task 7.2: Wire diagnostics into runner.py

**Files:**
- Modify: `src/backtest/runner.py` (end-of-backtest, after report card)

After `BacktestReportCard` is built (~line 830), if diagnostics enabled:

```python
if config.get("analytics", {}).get("diagnostics", {}).get("enabled", False):
    min_trades = config["analytics"]["diagnostics"].get("min_trades_for_analysis", 20)
    trades_dicts = [
        {
            "strategy": t.strategy,
            "pnl": t.pnl,
            "regime_trend": t.metadata.get("regime_trend", "UNKNOWN"),
            "entry_hour": t.entry_time.hour,
            "hold_minutes": t.hold_minutes,
            "exit_type": t.metadata.get("exit_type", "unknown"),
        }
        for t in trade_records
    ]
    diag = run_diagnostics(trades_dicts, min_trades=min_trades)
    report.diagnostics = diag
    logger.info("diagnostics_summary", summary=diag.summary)
```

**Commit:**

```bash
git commit -m "feat(backtest): integrate diagnostics engine into backtest pipeline"
```

---

## Phase 8: Results Persistence & API

### Task 8.1: JSON result serialization

**Files:**
- Create: `src/backtest/result_store.py`
- Test: `tests/unit/test_result_store.py`

**Step 1: Implement a `save_backtest_result()` function** that:
- Takes a `BacktestReportCard` (with all analytics)
- Generates a run_id (hash of config + date range)
- Serializes to JSON in `results/{run_id}.json`
- Includes: metrics, equity curve, trade records, monte carlo, benchmark, diagnostics

**Step 2: Implement `load_backtest_result(run_id)` and `list_backtest_runs()`.

**Step 3: Wire `save_backtest_result()` into runner.py at end of backtest.

**Commit:**

```bash
git commit -m "feat(backtest): add JSON result persistence for dashboard API"
```

---

### Task 8.2: FastAPI backtest API

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/backtest_api.py`
- Test: `tests/unit/test_backtest_api.py`

**Step 1: Create FastAPI app with endpoints:**

```python
# src/api/backtest_api.py
from fastapi import FastAPI, HTTPException
from src.backtest.result_store import list_backtest_runs, load_backtest_result

app = FastAPI(title="Cerberus Backtest API")

@app.get("/api/backtest/runs")
def get_runs():
    return list_backtest_runs()

@app.get("/api/backtest/runs/{run_id}/equity")
def get_equity(run_id: str):
    result = load_backtest_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return {"equity_curve": result["equity_curve"], "benchmark": result.get("benchmark")}

@app.get("/api/backtest/runs/{run_id}/trades")
def get_trades(run_id: str):
    result = load_backtest_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return {"trades": result["trades"]}

@app.get("/api/backtest/runs/{run_id}/monte-carlo")
def get_monte_carlo(run_id: str):
    result = load_backtest_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return result.get("monte_carlo", {})

@app.get("/api/backtest/runs/{run_id}/regime-splits")
def get_regime_splits(run_id: str):
    result = load_backtest_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return result.get("diagnostics", {}).get("regime_mismatches", [])

@app.get("/api/wfo/runs/{run_id}/sensitivity")
def get_sensitivity(run_id: str):
    result = load_backtest_result(run_id)
    if not result:
        raise HTTPException(404, "Run not found")
    return result.get("param_sensitivity", {})
```

**Step 2: Test with httpx TestClient**

```python
# tests/unit/test_backtest_api.py
import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_list_runs_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("src.backtest.result_store.RESULTS_DIR", tmp_path)
    from src.api.backtest_api import app
    client = TestClient(app)
    resp = client.get("/api/backtest/runs")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.unit
def test_get_run_not_found():
    from src.api.backtest_api import app
    client = TestClient(app)
    resp = client.get("/api/backtest/runs/nonexistent/equity")
    assert resp.status_code == 404
```

**Step 3: Commit**

```bash
git commit -m "feat(api): add FastAPI backtest API for dashboard consumption"
```

---

## Phase 9: EmpireUI Dashboard Views

### Task 9.1: API hooks and types

**Files:**
- Modify: `EmpireUI/src/features/backtests/types.ts` (extend types)
- Modify: `EmpireUI/src/features/backtests/hooks/useBacktestData.ts` (add new query hooks)

**Step 1:** Add TypeScript types for Monte Carlo, benchmark, diagnostics, trades, regime splits.

**Step 2:** Add TanStack Query hooks:
- `useBacktestEquity(runId)` → fetches `/api/backtest/runs/{id}/equity`
- `useBacktestTrades(runId)` → fetches `/api/backtest/runs/{id}/trades`
- `useBacktestMonteCarlo(runId)` → fetches `/api/backtest/runs/{id}/monte-carlo`
- `useBacktestRegimeSplits(runId)` → fetches `/api/backtest/runs/{id}/regime-splits`

**Step 3: Commit**

```bash
cd /Users/jacobmcmillan/Empire/EmpireUI
git add src/features/backtests/types.ts src/features/backtests/hooks/useBacktestData.ts
git commit -m "feat(ui): add API hooks for enhanced backtest analytics"
```

---

### Task 9.2: Equity Curve Overlay component

**Files:**
- Create: `EmpireUI/src/features/backtests/components/EquityCurveOverlay.tsx`
- Modify: `EmpireUI/src/features/backtests/BacktestDetail.tsx` (integrate)

**Step 1:** Build Recharts `ComposedChart` with:
- Strategy equity as `<Line>` (primary color)
- SPY benchmark as `<Line>` (gray, dashed)
- Monte Carlo 5th/95th bands as `<Area>` (transparent fill)
- WFO period shading via `<ReferenceArea>` (blue IS, green OOS, yellow holdout)
- Drawdown subplot as separate `<AreaChart>` below

**Step 2:** Integrate into `BacktestDetail.tsx` — replace or enhance existing equity chart.

**Step 3: Commit**

```bash
git commit -m "feat(ui): add equity curve overlay with benchmark and Monte Carlo bands"
```

---

### Task 9.3: Trade Scatter Plot component

**Files:**
- Create: `EmpireUI/src/features/backtests/components/TradeScatterPlot.tsx`
- Modify: `EmpireUI/src/features/backtests/BacktestDetail.tsx`

**Step 1:** Build two Recharts `ScatterChart`s:
- Hold duration (X) vs PnL (Y), colored by strategy
- Entry time (X) vs PnL (Y), colored by win/loss
- Add filter controls (strategy dropdown, date range)
- Click handler → detail popover

**Step 2:** Add to BacktestDetail page as a new tab or section.

**Step 3: Commit**

```bash
git commit -m "feat(ui): add trade scatter plot with strategy and time-of-day views"
```

---

### Task 9.4: Regime Performance Breakdown component

**Files:**
- Create: `EmpireUI/src/features/backtests/components/RegimeBreakdown.tsx`
- Modify: `EmpireUI/src/features/backtests/BacktestDetail.tsx`

**Step 1:** Build:
- Grouped `<BarChart>` per regime axis (trend/vol/session)
- Each group shows avg PnL per strategy per regime state
- Color-coded: green positive, red negative
- Trade count labels on bars

**Step 2:** Integrate into BacktestDetail.

**Step 3: Commit**

```bash
git commit -m "feat(ui): add regime performance breakdown charts"
```

---

## Phase Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|-----------------|
| **1. Fill Model** | 1.1–1.5 | FillModel protocol, Fixed + VolumeAware implementations, executor integration |
| **2. Overnight** | 2.1–2.3 | Per-strategy overnight config, per-position EOD flatten, strategy ref on Position |
| **3. Data Quality** | 3.1–3.2 | Pre-backtest data checker, runner integration |
| **4. Benchmark** | 4.1–4.2 | Alpha/beta/capture ratios, report integration |
| **5. Monte Carlo** | 5.1–5.2 | Bootstrap simulation, confidence intervals, report integration |
| **6. WFO** | 6.1–6.3 | Automated holdout, parameter sensitivity, WFO integration |
| **7. Diagnostics** | 7.1–7.2 | Regime mismatch, time edge, hold analysis, runner integration |
| **8. API** | 8.1–8.2 | JSON result store, FastAPI endpoints |
| **9. Dashboard** | 9.1–9.4 | API hooks, equity overlay, trade scatter, regime breakdown |

**Dependency order:** Phase 1-5 are independent and can be parallelized. Phase 6 depends on Phase 5 (Monte Carlo in WFO). Phase 7 depends on Phases 3-4 (regime data, benchmark). Phase 8 depends on all Cerberus phases. Phase 9 depends on Phase 8.
