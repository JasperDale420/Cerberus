#!/usr/bin/env python
"""DEPRECATED — use scripts/cerberus_autoresearch.py instead.

This script previously implemented a 5-window cross-validated variant of the
deprecated weighted-average composite score (see autoresearch_score.py header).
The active honest scorer lives in scripts/cerberus_autoresearch.py.
"""

import sys


def main() -> int:
    sys.stderr.write(
        "DEPRECATED: scripts/autoresearch_score_cv.py was the pre-2026-04-25 "
        "weighted-average scorer (CV variant). Use scripts/cerberus_autoresearch.py "
        "(invoked by scripts/autoresearch_driver.sh) instead. "
        "See debug/260430-1728-autoresearch-loop/findings.md M3.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
