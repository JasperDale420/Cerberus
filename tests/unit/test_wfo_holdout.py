import pytest


@pytest.mark.unit
def test_holdout_result_structure():
    from src.analytics.optuna_harness import HoldoutResult

    result = HoldoutResult(
        params_used={"stop_atr_mult": 2.0},
        holdout_sharpe=1.5,
        holdout_pf=2.0,
        holdout_max_dd=8.5,
        holdout_n_trades=25,
        holdout_score=0.75,
        oos_to_holdout_ratio=0.85,
        passed=True,
    )
    assert result.passed is True
    assert result.oos_to_holdout_ratio == 0.85


@pytest.mark.unit
def test_holdout_fails_when_ratio_below_threshold():
    from src.analytics.optuna_harness import HoldoutResult

    result = HoldoutResult(
        params_used={},
        holdout_sharpe=0.3,
        holdout_pf=0.8,
        holdout_max_dd=25.0,
        holdout_n_trades=10,
        holdout_score=0.2,
        oos_to_holdout_ratio=0.3,
        passed=False,
    )
    assert result.passed is False
