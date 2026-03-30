"""Tests for src/analytics/regime_stats.py — per-regime performance breakdowns."""

from __future__ import annotations

import pytest

from src.analytics.regime_stats import (
    compute_enrichment_breakdown,
    compute_regime_breakdown,
    compute_regime_matrix,
    compute_regime_transitions,
    format_regime_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trades() -> list[dict]:
    """Sample trades with regime enrichment fields."""
    return [
        {
            "pnl": 10.0,
            "entry_regime_trend": "UP",
            "entry_regime_vol": "LOW",
            "entry_liquidity": "GOOD",
            "entry_correlation": "HIGH",
            "entry_near_earnings": False,
            "entry_near_fomc": False,
            "entry_opex_week": False,
            "entry_session_phase": "OPENING",
        },
        {
            "pnl": -5.0,
            "entry_regime_trend": "UP",
            "entry_regime_vol": "NORMAL",
            "entry_liquidity": "GOOD",
            "entry_correlation": "HIGH",
            "entry_near_earnings": True,
            "entry_near_fomc": False,
            "entry_opex_week": False,
            "entry_session_phase": "MIDDAY",
        },
        {
            "pnl": 20.0,
            "entry_regime_trend": "DOWN",
            "entry_regime_vol": "HIGH",
            "entry_liquidity": "THIN",
            "entry_correlation": "LOW",
            "entry_near_earnings": False,
            "entry_near_fomc": True,
            "entry_opex_week": True,
            "entry_session_phase": "POWER_HOUR",
        },
        {
            "pnl": -3.0,
            "entry_regime_trend": "FLAT",
            "entry_regime_vol": "NORMAL",
            "entry_liquidity": "GOOD",
            "entry_correlation": "HIGH",
            "entry_near_earnings": False,
            "entry_near_fomc": False,
            "entry_opex_week": False,
            "entry_session_phase": "OPENING",
        },
        {
            "pnl": 8.0,
            "entry_regime_trend": "UP",
            "entry_regime_vol": "LOW",
            "entry_liquidity": "GOOD",
            "entry_correlation": "HIGH",
            "entry_near_earnings": False,
            "entry_near_fomc": False,
            "entry_opex_week": True,
            "entry_session_phase": "MIDDAY",
        },
    ]


def _make_trades_no_regime() -> list[dict]:
    """Trades without any regime fields (backward compat)."""
    return [
        {"pnl": 10.0, "symbol": "AAPL", "strategy": "mean_reversion"},
        {"pnl": -5.0, "symbol": "MSFT", "strategy": "trend_rider"},
        {"pnl": 3.0, "symbol": "AAPL", "strategy": "mean_reversion"},
    ]


# ---------------------------------------------------------------------------
# compute_regime_breakdown
# ---------------------------------------------------------------------------


class TestComputeRegimeBreakdown:
    def test_groups_by_trend(self):
        trades = _make_trades()
        result = compute_regime_breakdown(trades, regime_field="entry_regime_trend")

        assert "UP" in result
        assert "DOWN" in result
        assert "FLAT" in result

        # UP has 3 trades: +10, -5, +8
        assert result["UP"]["n_trades"] == 3
        assert result["UP"]["total_pnl"] == pytest.approx(13.0, abs=0.01)
        assert result["UP"]["win_rate"] == pytest.approx(2 / 3, abs=0.01)

        # DOWN has 1 trade: +20
        assert result["DOWN"]["n_trades"] == 1
        assert result["DOWN"]["win_rate"] == pytest.approx(1.0)

        # FLAT has 1 trade: -3
        assert result["FLAT"]["n_trades"] == 1
        assert result["FLAT"]["win_rate"] == pytest.approx(0.0)

    def test_groups_by_volatility(self):
        trades = _make_trades()
        result = compute_regime_breakdown(trades, regime_field="entry_regime_vol")

        assert "LOW" in result
        assert "NORMAL" in result
        assert "HIGH" in result

        assert result["LOW"]["n_trades"] == 2
        assert result["NORMAL"]["n_trades"] == 2
        assert result["HIGH"]["n_trades"] == 1

    def test_empty_trades(self):
        result = compute_regime_breakdown([])
        assert result == {}

    def test_missing_field_returns_empty(self):
        trades = _make_trades_no_regime()
        result = compute_regime_breakdown(trades, regime_field="entry_regime_trend")
        assert result == {}

    def test_profit_factor_computed(self):
        trades = _make_trades()
        result = compute_regime_breakdown(trades, regime_field="entry_regime_trend")

        # UP: gross_profit=18, gross_loss=5 => PF=3.6
        assert result["UP"]["profit_factor"] == pytest.approx(3.6, abs=0.01)

    def test_custom_pnl_key(self):
        trades = [{"pnl_r": 5.0, "entry_regime_trend": "UP"}, {"pnl_r": -2.0, "entry_regime_trend": "UP"}]
        result = compute_regime_breakdown(trades, regime_field="entry_regime_trend", pnl_key="pnl_r")
        assert result["UP"]["n_trades"] == 2
        assert result["UP"]["total_pnl"] == pytest.approx(3.0, abs=0.01)


# ---------------------------------------------------------------------------
# compute_regime_matrix
# ---------------------------------------------------------------------------


class TestComputeRegimeMatrix:
    def test_matrix_structure(self):
        trades = _make_trades()
        result = compute_regime_matrix(trades)

        assert "rows" in result
        assert "cols" in result
        assert "cells" in result

        assert sorted(result["rows"]) == ["DOWN", "FLAT", "UP"]
        assert sorted(result["cols"]) == ["HIGH", "LOW", "NORMAL"]

    def test_cell_values(self):
        trades = _make_trades()
        result = compute_regime_matrix(trades)

        # UP|LOW: trades with pnl +10, +8
        cell = result["cells"]["UP|LOW"]
        assert cell["n_trades"] == 2
        assert cell["win_rate"] == pytest.approx(1.0)
        assert cell["avg_pnl"] == pytest.approx(9.0, abs=0.01)

        # UP|NORMAL: trade with pnl -5
        cell = result["cells"]["UP|NORMAL"]
        assert cell["n_trades"] == 1
        assert cell["win_rate"] == pytest.approx(0.0)

        # DOWN|HIGH: trade with pnl +20
        cell = result["cells"]["DOWN|HIGH"]
        assert cell["n_trades"] == 1
        assert cell["avg_pnl"] == pytest.approx(20.0)

    def test_empty_trades(self):
        result = compute_regime_matrix([])
        assert result == {}

    def test_missing_fields_returns_empty(self):
        trades = _make_trades_no_regime()
        result = compute_regime_matrix(trades)
        assert result == {}

    def test_partial_fields_skipped(self):
        """Trades missing one of the two fields are excluded."""
        trades = [
            {"pnl": 5.0, "entry_regime_trend": "UP"},  # missing vol
            {"pnl": 3.0, "entry_regime_vol": "LOW"},  # missing trend
            {"pnl": 10.0, "entry_regime_trend": "UP", "entry_regime_vol": "LOW"},
        ]
        result = compute_regime_matrix(trades)
        assert result["cells"]["UP|LOW"]["n_trades"] == 1


# ---------------------------------------------------------------------------
# compute_enrichment_breakdown
# ---------------------------------------------------------------------------


class TestComputeEnrichmentBreakdown:
    def test_boolean_fields(self):
        trades = _make_trades()
        result = compute_enrichment_breakdown(trades)

        # near_earnings: 1 True, 4 False
        assert "near_earnings" in result
        assert result["near_earnings"]["True"]["n_trades"] == 1
        assert result["near_earnings"]["False"]["n_trades"] == 4

        # near_fomc: 1 True, 4 False
        assert "near_fomc" in result
        assert result["near_fomc"]["True"]["n_trades"] == 1

        # opex_week: 2 True, 3 False
        assert "opex_week" in result
        assert result["opex_week"]["True"]["n_trades"] == 2
        assert result["opex_week"]["False"]["n_trades"] == 3

    def test_session_phase(self):
        trades = _make_trades()
        result = compute_enrichment_breakdown(trades)

        assert "session_phase" in result
        assert "OPENING" in result["session_phase"]
        assert result["session_phase"]["OPENING"]["n_trades"] == 2

    def test_liquidity(self):
        trades = _make_trades()
        result = compute_enrichment_breakdown(trades)

        assert "liquidity" in result
        assert "GOOD" in result["liquidity"]
        assert result["liquidity"]["GOOD"]["n_trades"] == 4

    def test_missing_fields_returns_empty(self):
        trades = _make_trades_no_regime()
        result = compute_enrichment_breakdown(trades)
        assert result == {}


# ---------------------------------------------------------------------------
# format_regime_report
# ---------------------------------------------------------------------------


class TestFormatRegimeReport:
    def test_empty_data(self):
        result = format_regime_report({})
        assert "No regime data" in result

    def test_formats_group_stats(self):
        breakdowns = {
            "by_trend": {
                "UP": {"n_trades": 3, "win_rate": 0.67, "profit_factor": 3.6, "avg_pnl": 4.33, "total_pnl": 13.0},
                "DOWN": {"n_trades": 1, "win_rate": 1.0, "profit_factor": 0.0, "avg_pnl": 20.0, "total_pnl": 20.0},
            }
        }
        result = format_regime_report(breakdowns)
        assert "by_trend" in result
        assert "UP" in result
        assert "DOWN" in result

    def test_formats_matrix(self):
        matrix = {
            "rows": ["UP", "DOWN"],
            "cols": ["LOW", "HIGH"],
            "cells": {
                "UP|LOW": {"n_trades": 2, "win_rate": 1.0, "avg_pnl": 9.0},
                "DOWN|HIGH": {"n_trades": 1, "win_rate": 1.0, "avg_pnl": 20.0},
            },
        }
        breakdowns = {"trend_x_vol_matrix": matrix}
        result = format_regime_report(breakdowns)
        assert "trend_x_vol_matrix" in result
        assert "LOW" in result
        assert "HIGH" in result


# ---------------------------------------------------------------------------
# Integration: backward compat
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Ensure everything degrades gracefully when regime fields are absent."""

    def test_all_functions_with_no_regime_trades(self):
        trades = _make_trades_no_regime()

        assert compute_regime_breakdown(trades) == {}
        assert compute_regime_matrix(trades) == {}
        assert compute_enrichment_breakdown(trades) == {}

    def test_mixed_trades_some_with_regime(self):
        """Some trades have regime fields, some don't."""
        trades = [
            {"pnl": 10.0, "entry_regime_trend": "UP"},
            {"pnl": -5.0},  # no regime field
            {"pnl": 3.0, "entry_regime_trend": "UP"},
        ]
        result = compute_regime_breakdown(trades, regime_field="entry_regime_trend")
        # Only 2 trades have the field
        assert result["UP"]["n_trades"] == 2
        assert result["UP"]["total_pnl"] == pytest.approx(13.0, abs=0.01)


# ---------------------------------------------------------------------------
# compute_regime_transitions
# ---------------------------------------------------------------------------


def _make_transition_trades() -> list[dict]:
    """Trades with both entry and exit regime fields for transition analysis."""
    return [
        # Stable trade: UP/LOW -> UP/LOW
        {
            "pnl": 15.0,
            "side": "buy",
            "entry_regime_trend": "UP",
            "entry_regime_vol": "LOW",
            "exit_regime_trend": "UP",
            "exit_regime_vol": "LOW",
        },
        # Trend reversal: UP -> DOWN (adverse)
        {
            "pnl": -20.0,
            "side": "buy",
            "entry_regime_trend": "UP",
            "entry_regime_vol": "NORMAL",
            "exit_regime_trend": "DOWN",
            "exit_regime_vol": "NORMAL",
        },
        # Vol expansion on a short (adverse: vol_expansion_short)
        {
            "pnl": -30.0,
            "side": "sell",
            "entry_regime_trend": "FLAT",
            "entry_regime_vol": "LOW",
            "exit_regime_trend": "FLAT",
            "exit_regime_vol": "HIGH",
        },
        # Vol expansion on a long (adverse: vol_expansion_long)
        {
            "pnl": 5.0,
            "side": "buy",
            "entry_regime_trend": "UP",
            "entry_regime_vol": "NORMAL",
            "exit_regime_trend": "UP",
            "exit_regime_vol": "SHOCK",
        },
        # Stable trade: DOWN/HIGH -> DOWN/HIGH
        {
            "pnl": 10.0,
            "side": "sell",
            "entry_regime_trend": "DOWN",
            "entry_regime_vol": "HIGH",
            "exit_regime_trend": "DOWN",
            "exit_regime_vol": "HIGH",
        },
        # Trend reversal: DOWN -> UP
        {
            "pnl": -8.0,
            "side": "sell",
            "entry_regime_trend": "DOWN",
            "entry_regime_vol": "NORMAL",
            "exit_regime_trend": "UP",
            "exit_regime_vol": "NORMAL",
        },
    ]


class TestComputeRegimeTransitions:
    def test_transition_counts(self):
        trades = _make_transition_trades()
        result = compute_regime_transitions(trades)

        assert result["total_trades"] == 6
        # Stable: trade 0 (UP/LOW->UP/LOW) and trade 4 (DOWN/HIGH->DOWN/HIGH) = 2
        assert result["same_regime"] == 2
        assert result["regime_changed"] == 4

    def test_transition_matrix(self):
        trades = _make_transition_trades()
        result = compute_regime_transitions(trades)
        matrix = result["transition_matrix"]

        # UP->UP: trades 0 (+15) and 3 (+5) = 2 trades
        assert "UP->UP" in matrix
        assert matrix["UP->UP"]["count"] == 2
        assert matrix["UP->UP"]["avg_pnl"] == pytest.approx(10.0, abs=0.01)
        assert matrix["UP->UP"]["win_rate"] == pytest.approx(1.0)

        # UP->DOWN: trade 1 (-20) = 1 trade
        assert "UP->DOWN" in matrix
        assert matrix["UP->DOWN"]["count"] == 1
        assert matrix["UP->DOWN"]["avg_pnl"] == pytest.approx(-20.0, abs=0.01)
        assert matrix["UP->DOWN"]["win_rate"] == pytest.approx(0.0)

        # FLAT->FLAT: trade 2 (-30)
        assert "FLAT->FLAT" in matrix
        assert matrix["FLAT->FLAT"]["count"] == 1

        # DOWN->DOWN: trade 4 (+10)
        assert "DOWN->DOWN" in matrix
        assert matrix["DOWN->DOWN"]["count"] == 1

        # DOWN->UP: trade 5 (-8)
        assert "DOWN->UP" in matrix
        assert matrix["DOWN->UP"]["count"] == 1

    def test_adverse_trend_reversal(self):
        trades = _make_transition_trades()
        result = compute_regime_transitions(trades)
        adverse = {a["type"]: a for a in result["adverse_transitions"]}

        assert "trend_reversal" in adverse
        # Trades 1 (UP->DOWN, -20) and 5 (DOWN->UP, -8) = 2
        assert adverse["trend_reversal"]["count"] == 2
        assert adverse["trend_reversal"]["avg_pnl"] == pytest.approx(-14.0, abs=0.01)

    def test_adverse_vol_expansion_short(self):
        trades = _make_transition_trades()
        result = compute_regime_transitions(trades)
        adverse = {a["type"]: a for a in result["adverse_transitions"]}

        assert "vol_expansion_short" in adverse
        # Trade 2: sell, LOW->HIGH, pnl=-30
        assert adverse["vol_expansion_short"]["count"] == 1
        assert adverse["vol_expansion_short"]["avg_pnl"] == pytest.approx(-30.0, abs=0.01)

    def test_adverse_vol_expansion_long(self):
        trades = _make_transition_trades()
        result = compute_regime_transitions(trades)
        adverse = {a["type"]: a for a in result["adverse_transitions"]}

        assert "vol_expansion_long" in adverse
        # Trade 3: buy, NORMAL->SHOCK, pnl=+5
        assert adverse["vol_expansion_long"]["count"] == 1
        assert adverse["vol_expansion_long"]["avg_pnl"] == pytest.approx(5.0, abs=0.01)

    def test_adverse_liquidity_drain(self):
        trades = [
            {
                "pnl": -12.0,
                "side": "buy",
                "entry_regime_trend": "UP",
                "entry_regime_vol": "NORMAL",
                "exit_regime_trend": "UP",
                "exit_regime_vol": "NORMAL",
                "entry_liquidity": "GOOD",
                "exit_liquidity": "THIN",
            },
        ]
        result = compute_regime_transitions(trades)
        adverse = {a["type"]: a for a in result["adverse_transitions"]}

        assert "liquidity_drain" in adverse
        assert adverse["liquidity_drain"]["count"] == 1
        assert adverse["liquidity_drain"]["avg_pnl"] == pytest.approx(-12.0, abs=0.01)

    def test_pnl_by_stability(self):
        trades = _make_transition_trades()
        result = compute_regime_transitions(trades)
        stability = result["pnl_by_stability"]

        # Stable: trades 0 (+15) and 4 (+10) => avg=12.5
        assert stability["stable"]["n"] == 2
        assert stability["stable"]["avg_pnl"] == pytest.approx(12.5, abs=0.01)

        # Changed: trades 1 (-20), 2 (-30), 3 (+5), 5 (-8) => avg=-13.25
        assert stability["changed"]["n"] == 4
        assert stability["changed"]["avg_pnl"] == pytest.approx(-13.25, abs=0.01)

    def test_empty_trades(self):
        result = compute_regime_transitions([])
        assert result == {}

    def test_trades_without_regime_fields(self):
        """Backward compat: trades missing regime fields return empty."""
        trades = _make_trades_no_regime()
        result = compute_regime_transitions(trades)
        assert result == {}

    def test_partial_regime_fields_skipped(self):
        """Trades with only entry but no exit regime fields are skipped."""
        trades = [
            {
                "pnl": 10.0,
                "entry_regime_trend": "UP",
                "entry_regime_vol": "LOW",
                # missing exit fields
            },
            {
                "pnl": 5.0,
                "entry_regime_trend": "UP",
                "entry_regime_vol": "LOW",
                "exit_regime_trend": "UP",
                "exit_regime_vol": "LOW",
            },
        ]
        result = compute_regime_transitions(trades)
        assert result["total_trades"] == 1
        assert result["same_regime"] == 1

    def test_mixed_trades_with_and_without_fields(self):
        """Mix of trades with and without regime fields."""
        trades = [
            {"pnl": 10.0},  # no regime fields
            {
                "pnl": -5.0,
                "side": "buy",
                "entry_regime_trend": "UP",
                "entry_regime_vol": "LOW",
                "exit_regime_trend": "DOWN",
                "exit_regime_vol": "HIGH",
            },
        ]
        result = compute_regime_transitions(trades)
        assert result["total_trades"] == 1
        assert result["regime_changed"] == 1
