from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataRequirements:
    """Declares what data a strategy needs from the unified data client."""

    streams: list[str] = field(default_factory=lambda: ["bars"])
    on_scan: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)


def aggregate_requirements(reqs: list[DataRequirements]) -> DataRequirements:
    """Union all requirements from registered strategies."""
    streams: set[str] = set()
    on_scan: set[str] = set()
    indicators: set[str] = set()
    for r in reqs:
        streams.update(r.streams)
        on_scan.update(r.on_scan)
        indicators.update(r.indicators)
    return DataRequirements(
        streams=sorted(streams),
        on_scan=sorted(on_scan),
        indicators=sorted(indicators),
    )
