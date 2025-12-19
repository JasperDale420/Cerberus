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

        if (self.avg_loss or 0.0) == 0.0:
            self.value = 100.0
            return 100.0

        rs = float(self.avg_gain or 0.0) / float(self.avg_loss)
        self.value = float(100.0 - (100.0 / (1.0 + rs)))
        return float(self.value)
