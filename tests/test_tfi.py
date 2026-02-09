import pytest

from src.data.calculator import FeatureCalculator


def test_calculate_tfi_empty():
    calc = FeatureCalculator()
    assert calc.calculate_tfi([]) == 0.0


def test_calculate_tfi_basic_buy():
    calc = FeatureCalculator()
    trades = [
        {"p": 100.0, "s": 100},
        {"p": 100.1, "s": 100},  # Uptick
    ]
    # TFI should be 100/200 = 0.5? No, tick test.
    # 1st trade: no uptick/downtick yet. last_price=100.
    # 2nd trade: 100.1 > 100 -> Buy. BuyVol = 100. TotalVol = 200.
    # (100 - 0) / 200 = 0.5.
    assert calc.calculate_tfi(trades) == 0.5


def test_calculate_tfi_lee_ready_tick_test():
    calc = FeatureCalculator()
    trades = [
        {"p": 100.0, "s": 10},
        {"p": 101.0, "s": 20},  # Buy (Uptick)
        {"p": 101.0, "s": 10},  # Buy (Repeat Tick)
        {"p": 99.0, "s": 30},  # Sell (Downtick)
        {"p": 99.0, "s": 10},  # Sell (Repeat Tick)
    ]
    # Buy Vol: 20 (uptick) + 10 (repeat) = 30
    # Sell Vol: 30 (downtick) + 10 (repeat) = 40
    # Total Vol: 10 + 20 + 10 + 30 + 10 = 80
    # TFI = (30 - 40) / 80 = -10 / 80 = -0.125
    assert calc.calculate_tfi(trades) == -0.125


def test_calculate_tfi_all_buys():
    calc = FeatureCalculator()
    trades = [
        {"p": 100.0, "s": 10},
        {"p": 101.0, "s": 10},
        {"p": 102.0, "s": 10},
    ]
    # (10 + 10) / 30 = 0.666...
    assert calc.calculate_tfi(trades) == pytest.approx(0.6666666666666666)


def test_calculate_tfi_all_sells():
    calc = FeatureCalculator()
    trades = [
        {"p": 100.0, "s": 10},
        {"p": 99.0, "s": 10},
        {"p": 98.0, "s": 10},
    ]
    # (0 - 20) / 30 = -0.666...
    assert calc.calculate_tfi(trades) == pytest.approx(-0.6666666666666666)
