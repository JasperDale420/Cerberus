import numpy as np
import pandas as pd
import pytest

from src.config.models import PairTradingConfig
from src.core.logger import StructuredLogger
from src.scanner.pair_scanner import PairScanner


@pytest.fixture
def pair_scanner():
    cfg = PairTradingConfig(enabled=True, min_correlation=0.5, max_eg_pvalue=0.1, min_half_life=0)
    logger = StructuredLogger("test")
    return PairScanner(cfg, logger)


def test_find_pairs_cointegrated(pair_scanner):
    # Generate cleaner cointegrated data
    np.random.seed(42)
    n = 250
    x = np.cumsum(np.random.randn(n)) + 100
    # y is locked to x with very small noise
    y = 0.8 * x + np.random.randn(n) * 0.1 + 10

    df = pd.DataFrame({"SYM1": x, "SYM2": y, "SYM3": np.random.randn(n) + 100})

    # Check coint pvalue manually for debug
    from statsmodels.tsa.stattools import coint

    _, pval, _ = coint(df["SYM1"], df["SYM2"])
    print(f"Manual coint pvalue (1,2): {pval}")

    pairs = pair_scanner.find_pairs(df)
    print(f"Pairs found: {[(p['symbol_a'], p['symbol_b'], p['pvalue']) for p in pairs]}")

    assert len(pairs) >= 1
    found = any(
        (p["symbol_a"] == "SYM1" and p["symbol_b"] == "SYM2") or (p["symbol_a"] == "SYM2" and p["symbol_b"] == "SYM1")
        for p in pairs
    )
    assert found


def test_calculate_half_life(pair_scanner):
    # Ornstein-Uhlenbeck process with strong mean reversion
    n = 500
    theta = 0.5  # speed of reversion
    mu = 0.0  # mean
    spread = [10.0]
    for _ in range(n - 1):
        spread.append(spread[-1] + theta * (mu - spread[-1]) + np.random.randn())

    s = pd.Series(spread)
    half_life = pair_scanner._calculate_half_life(s)

    # half-life = log(2) / theta approx 0.693 / 0.5 approx 1.38
    assert 0.5 < half_life < 5.0


def test_find_pairs_disabled():
    cfg = PairTradingConfig(enabled=False)
    logger = StructuredLogger("test")
    scanner = PairScanner(cfg, logger)

    df = pd.DataFrame({"A": [1, 2, 3], "B": [1.1, 2.1, 3.1]})
    assert scanner.find_pairs(df) == []
