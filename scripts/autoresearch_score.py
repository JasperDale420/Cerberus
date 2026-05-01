#!/usr/bin/env python
"""DEPRECATED — use scripts/cerberus_autoresearch.py instead.

This script previously implemented a weighted-average composite score
(0.30*pnl + 0.25*sharpe + 0.20*pf + 0.15*wr + 0.10*trade_score) which
proved gameable: a strategy losing money in 12 of 13 OOS windows could
still register a "best score" by averaging one lucky window.

The honest replacement (composite = ratio_vs_spy with hard gates) lives in
`scripts/cerberus_autoresearch.py`. The autoresearch driver
(`scripts/autoresearch_driver.sh`) calls that scorer directly.

This stub exists to prevent accidental invocation from older docs or
muscle memory.
"""

import sys


def main() -> int:
    sys.stderr.write(
        "DEPRECATED: scripts/autoresearch_score.py was the pre-2026-04-25 "
        "weighted-average scorer. Use scripts/cerberus_autoresearch.py "
        "(invoked by scripts/autoresearch_driver.sh) instead. "
        "See debug/260430-1728-autoresearch-loop/findings.md M3.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
