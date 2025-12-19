from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Protocol

from src.core.domain import Bar


class BarsProvider(Protocol):
    def get_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str
    ) -> List[Bar]:
        """
        Return bars for `symbol` in [start, end], inclusive, in ascending time order.
        """
        ...


@dataclass(frozen=True)
class JsonlBarsProvider:
    """
    Deterministic offline provider reading `*.jsonl` files.

    File naming convention: `{SYMBOL}_{timeframe}.jsonl` (e.g., `AAPL_1Min.jsonl`)
    Record format (one JSON object per line):
      {"t": "2025-01-01T14:30:00+00:00", "o": 1, "h": 1, "l": 1, "c": 1, "v": 100}
    """

    bars_dir: Path

    def get_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str
    ) -> List[Bar]:
        sym = str(symbol).strip().upper()
        tf = str(timeframe).strip()
        p = self.bars_dir / f"{sym}_{tf}.jsonl"
        if not p.exists():
            return []

        out: List[Bar] = []
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get("t") or row.get("timestamp")
            if not isinstance(t, str) or not t:
                continue
            ts = datetime.fromisoformat(t.replace("Z", "+00:00"))
            if ts < start or ts > end:
                continue

            out.append(
                Bar(
                    symbol=sym,
                    time=ts,
                    open=float(row.get("o") or row.get("open") or 0.0),
                    high=float(row.get("h") or row.get("high") or 0.0),
                    low=float(row.get("l") or row.get("low") or 0.0),
                    close=float(row.get("c") or row.get("close") or 0.0),
                    volume=float(row.get("v") or row.get("volume") or 0.0),
                )
            )

        out.sort(key=lambda b: b.time)
        return out
