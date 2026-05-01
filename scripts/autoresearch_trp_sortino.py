#!/usr/bin/env python
"""DEPRECATED — use scripts/cerberus_autoresearch.py instead.

This script previously implemented a Sortino+PnL composite scorer for
trend_rider_pro tuning. The active honest scorer lives in
scripts/cerberus_autoresearch.py and works for any strategy.
"""

import sys


def main() -> int:
    sys.stderr.write(
        "DEPRECATED: scripts/autoresearch_trp_sortino.py was the pre-2026-04-25 "
        "Sortino+PnL composite scorer. Use scripts/cerberus_autoresearch.py "
        "(invoked by scripts/autoresearch_driver.sh) instead. "
        "See debug/260430-1728-autoresearch-loop/findings.md M3.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
