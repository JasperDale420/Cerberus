from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class RollingSMA:
    window: int
    values: Deque[float]
    sum: float = 0.0
    value: Optional[float] = None
    prev_value: Optional[float] = None

    @classmethod
    def create(cls, window: int) -> "RollingSMA":
        w = max(1, int(window))
        return cls(window=w, values=deque(maxlen=w))

    def update(self, x: float) -> float:
        x = float(x)
        self.prev_value = self.value
        if len(self.values) == self.values.maxlen:
            self.sum -= float(self.values[0])
        self.values.append(x)
        self.sum += x
        self.value = self.sum / max(1, len(self.values))
        return float(self.value)


@dataclass
class RollingEMA:
    alpha: float
    value: Optional[float] = None
    prev_value: Optional[float] = None

    @classmethod
    def from_period(cls, period: int) -> "RollingEMA":
        p = max(1, int(period))
        return cls(alpha=2.0 / (p + 1.0))

    def update(self, x: float) -> float:
        x = float(x)
        self.prev_value = self.value
        if self.value is None:
            self.value = x
        else:
            self.value = (self.alpha * x) + ((1.0 - self.alpha) * self.value)
        return float(self.value)


@dataclass
class RollingStd:
    window: int
    values: Deque[float]
    sum: float = 0.0
    sumsq: float = 0.0

    @classmethod
    def create(cls, window: int) -> "RollingStd":
        w = max(1, int(window))
        return cls(window=w, values=deque(maxlen=w))

    def update(self, x: float) -> Tuple[float, float]:
        x = float(x)
        if len(self.values) == self.values.maxlen:
            old = float(self.values[0])
            self.sum -= old
            self.sumsq -= old * old
        self.values.append(x)
        self.sum += x
        self.sumsq += x * x
        n = max(1, len(self.values))
        mean = self.sum / n
        # population variance (deterministic, stable)
        var = max(0.0, (self.sumsq / n) - (mean * mean))
        return float(mean), float(var**0.5)


@dataclass
class RollingRSI:
    period: int
    prev_close: Optional[float] = None
    avg_gain: Optional[float] = None
    avg_loss: Optional[float] = None
    value: Optional[float] = None
    prev_value: Optional[float] = None

    def update(self, close: float) -> Optional[float]:
        close = float(close)
        self.prev_value = self.value
        if self.prev_close is None:
            self.prev_close = close
            self.value = None
            return None

        change = close - self.prev_close
        gain = max(0.0, change)
        loss = max(0.0, -change)

        p = max(1, int(self.period))
        if self.avg_gain is None or self.avg_loss is None:
            # deterministic seed with the first observed change
            self.avg_gain = gain
            self.avg_loss = loss
        else:
            self.avg_gain = ((self.avg_gain * (p - 1)) + gain) / p
            self.avg_loss = ((self.avg_loss * (p - 1)) + loss) / p

        self.prev_close = close

        if abs(self.avg_loss or 0.0) < 1e-9:
            self.value = 100.0
            return 100.0

        rs = float(self.avg_gain or 0.0) / float(self.avg_loss)
        self.value = float(100.0 - (100.0 / (1.0 + rs)))
        return float(self.value)


@dataclass
class RollingATR:
    """
    Incremental Average True Range using Wilder smoothing.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = Wilder-smoothed average of TR over the period.
    """

    period: int
    prev_close: Optional[float] = None
    atr: Optional[float] = None
    value: Optional[float] = None
    prev_value: Optional[float] = None
    count: int = 0

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        high, low, close = float(high), float(low), float(close)
        self.prev_value = self.value

        if self.prev_close is None:
            self.prev_close = close
            self.value = None
            return None

        # Calculate True Range
        tr = max(
            high - low,
            abs(high - self.prev_close),
            abs(low - self.prev_close),
        )

        p = max(1, int(self.period))
        self.count += 1

        if self.atr is None:
            # First TR value seeds the ATR
            self.atr = tr
        elif self.count < p:
            # Simple average until we have enough data
            self.atr = ((self.atr * (self.count - 1)) + tr) / self.count
        else:
            # Wilder smoothing: ATR = ((ATR * (p-1)) + TR) / p
            self.atr = ((self.atr * (p - 1)) + tr) / p

        self.prev_close = close
        self.value = self.atr
        return float(self.value)


@dataclass
class RollingADX:
    """
    Incremental Average Directional Index.

    Computes +DI, -DI, and ADX using Wilder smoothing.
    """

    period: int
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None
    prev_close: Optional[float] = None
    smoothed_tr: Optional[float] = None
    smoothed_plus_dm: Optional[float] = None
    smoothed_minus_dm: Optional[float] = None
    smoothed_dx: Optional[float] = None
    value: Optional[float] = None
    prev_value: Optional[float] = None
    count: int = 0

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        high, low, close = float(high), float(low), float(close)
        self.prev_value = self.value

        if self.prev_high is None or self.prev_low is None or self.prev_close is None:
            self.prev_high, self.prev_low, self.prev_close = high, low, close
            self.value = None
            return None

        # Calculate directional movements
        up_move = high - self.prev_high
        down_move = self.prev_low - low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0

        # True Range
        tr = max(
            high - low,
            abs(high - self.prev_close),
            abs(low - self.prev_close),
        )

        p = max(1, int(self.period))
        self.count += 1

        # Wilder smoothing for TR, +DM, -DM
        if self.smoothed_tr is None:
            self.smoothed_tr = tr
            self.smoothed_plus_dm = plus_dm
            self.smoothed_minus_dm = minus_dm
        else:
            self.smoothed_tr = self.smoothed_tr - (self.smoothed_tr / p) + tr
            if self.smoothed_plus_dm is not None:
                self.smoothed_plus_dm = self.smoothed_plus_dm - (self.smoothed_plus_dm / p) + plus_dm
            if self.smoothed_minus_dm is not None:
                self.smoothed_minus_dm = self.smoothed_minus_dm - (self.smoothed_minus_dm / p) + minus_dm

        # Calculate +DI and -DI
        if (
            self.smoothed_tr is not None
            and self.smoothed_tr > 0
            and self.smoothed_plus_dm is not None
            and self.smoothed_minus_dm is not None
        ):
            plus_di = 100.0 * self.smoothed_plus_dm / self.smoothed_tr
            minus_di = 100.0 * self.smoothed_minus_dm / self.smoothed_tr
        else:
            plus_di = 0.0
            minus_di = 0.0

        # Calculate DX
        di_sum = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0

        # Wilder smoothing for ADX
        if self.smoothed_dx is None:
            self.smoothed_dx = dx
        else:
            self.smoothed_dx = ((self.smoothed_dx * (p - 1)) + dx) / p

        self.prev_high, self.prev_low, self.prev_close = high, low, close
        self.value = self.smoothed_dx
        return float(self.value) if self.value is not None else None
