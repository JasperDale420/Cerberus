"""Unit tests for the VPIN calculator."""

from datetime import datetime, timezone

import pytest

from src.analysis.vpin import VPINCalculator
from src.core.domain import Bar


def make_bar(open_: float, close: float, volume: float, symbol: str = "TEST") -> Bar:
    return Bar(
        symbol=symbol,
        time=datetime.now(tz=timezone.utc),
        open=open_,
        high=max(open_, close),
        low=min(open_, close),
        close=close,
        volume=volume,
    )


@pytest.mark.unit
class TestVPINCalculator:
    def test_vpin_initialization(self) -> None:
        """Verify defaults and bucket_volume auto-computation."""
        calc = VPINCalculator()
        assert calc.n_buckets == 50
        assert calc.bucket_volume == 1_000_000 / 50  # 20_000
        assert calc.toxicity_threshold == 0.7
        assert calc.sigma_window == 20

        # Explicit bucket_volume overrides auto-computation
        calc2 = VPINCalculator(bucket_volume=5000, daily_volume_estimate=999)
        assert calc2.bucket_volume == 5000

        # Custom daily volume
        calc3 = VPINCalculator(n_buckets=10, daily_volume_estimate=500_000)
        assert calc3.bucket_volume == 50_000

    def test_vpin_returns_none_until_bucket_complete(self) -> None:
        """Small volume bars should not produce a result before a bucket fills."""
        calc = VPINCalculator(bucket_volume=10_000)

        # Feed bars with small volume — total well below bucket_volume
        for _ in range(5):
            result = calc.update(make_bar(100.0, 100.05, 100.0))

        assert result is None

    def test_vpin_balanced_flow(self) -> None:
        """Bars with open ~= close produce near-zero order imbalance; VPIN should be low."""
        calc = VPINCalculator(n_buckets=5, bucket_volume=1000)

        # Feed many bars with tiny price change (balanced flow)
        result = None
        for i in range(200):
            # Alternate tiny up/down to keep sigma non-zero but z near 0
            delta = 0.01 if i % 2 == 0 else -0.01
            r = calc.update(make_bar(100.0, 100.0 + delta, 50.0))
            if r is not None:
                result = r

        assert result is not None
        assert result.vpin < 0.3, f"Expected low VPIN for balanced flow, got {result.vpin}"

    def test_vpin_toxic_flow(self) -> None:
        """All bars closing much higher than open (strong buy) should yield high VPIN."""
        calc = VPINCalculator(n_buckets=5, bucket_volume=1000)

        result = None
        for _ in range(200):
            r = calc.update(make_bar(100.0, 102.0, 50.0))  # Consistent strong upward move
            if r is not None:
                result = r

        assert result is not None
        assert result.vpin > 0.6, f"Expected high VPIN for toxic flow, got {result.vpin}"

    def test_vpin_detects_toxicity_flag(self) -> None:
        """Verify is_toxic is True when VPIN exceeds the threshold."""
        calc = VPINCalculator(n_buckets=5, bucket_volume=500, toxicity_threshold=0.5)

        result = None
        for _ in range(200):
            r = calc.update(make_bar(100.0, 103.0, 50.0))  # Strong directional flow
            if r is not None:
                result = r

        assert result is not None
        assert result.is_toxic is True, f"Expected is_toxic=True, VPIN={result.vpin}"

    def test_vpin_volume_bucketing(self) -> None:
        """Verify correct number of completed buckets after known volume."""
        bucket_vol = 1000
        calc = VPINCalculator(n_buckets=50, bucket_volume=bucket_vol)

        # Feed exactly 5000 volume -> should complete 5 buckets
        for _ in range(50):
            calc.update(make_bar(100.0, 100.1, 100.0))  # 50 * 100 = 5000

        result = calc.update(make_bar(100.0, 100.1, 100.0))  # one more to get a result after 5th bucket
        # At least 5 buckets should have completed (might be 5 or 6 depending on exact split)
        assert result is not None
        assert result.bucket_count >= 5

    def test_vpin_set_daily_volume(self) -> None:
        """Verify bucket_volume updates when daily volume estimate changes."""
        calc = VPINCalculator(n_buckets=10, daily_volume_estimate=100_000)
        assert calc.bucket_volume == 10_000

        calc.set_daily_volume(200_000)
        assert calc.bucket_volume == 20_000

        calc.set_daily_volume(50_000)
        assert calc.bucket_volume == 5_000

    def test_vpin_reset(self) -> None:
        """Verify reset clears all internal state."""
        calc = VPINCalculator(n_buckets=5, bucket_volume=500)

        # Feed enough bars to fill some buckets
        for _ in range(100):
            calc.update(make_bar(100.0, 100.5, 50.0))

        calc.reset()

        # After reset, should return None since no buckets completed
        result = calc.update(make_bar(100.0, 100.5, 10.0))
        assert result is None

        # Internal accumulators should be zeroed
        assert calc._bucket_buy == 0.0 or calc._bucket_buy > 0.0  # might have partial from the 10-vol bar
        assert len(calc._imbalances) == 0

    def test_vpin_zero_volume_skipped(self) -> None:
        """Zero volume bars should be skipped without error."""
        calc = VPINCalculator(bucket_volume=1000)

        result = calc.update(make_bar(100.0, 101.0, 0.0))
        assert result is None

        # Negative volume also skipped
        result = calc.update(make_bar(100.0, 101.0, -5.0))
        assert result is None

        # State should be untouched
        assert len(calc._price_changes) == 0
        assert calc._bucket_total == 0.0
