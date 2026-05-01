#!/usr/bin/env python
"""DEPRECATED — use scripts/cerberus_autoresearch.py instead.

This script previously implemented the deprecated weighted-average composite
scorer (see autoresearch_score.py header) targeted at pair_trading_v2 over a
2020-2024 5-year window. The active honest scorer lives in
scripts/cerberus_autoresearch.py and accepts any strategy via dynamic import.
"""

import sys


def main() -> int:
    sys.stderr.write(
        "DEPRECATED: scripts/autoresearch_pairs_score.py was the pre-2026-04-25 "
        "weighted-average scorer for pair_trading_v2. Use scripts/cerberus_autoresearch.py "
        "(invoked by scripts/autoresearch_driver.sh) instead. "
        "See debug/260430-1728-autoresearch-loop/findings.md M3.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
