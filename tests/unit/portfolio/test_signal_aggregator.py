"""Tests for portfolio signal aggregator."""

from datetime import datetime, timezone

from src.core.domain import OrderSide, Signal
from src.portfolio.signal_aggregator import SignalAggregator


def _make_signal(
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    strategy: str = "momentum",
    entry: float = 150.0,
    stop: float = 145.0,
    target: float = 160.0,
) -> Signal:
    return Signal(
        symbol=symbol,
        side=side,
        size_hint=100,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        strategy=strategy,
        generated_at=datetime.now(timezone.utc),
    )


class TestSignalAggregatorBasic:
    """Core aggregation behaviour."""

    def test_single_signal_passes_through(self):
        agg = SignalAggregator()
        sig = _make_signal()
        result = agg.aggregate([sig])
        assert result == [sig]

    def test_empty_signals(self):
        agg = SignalAggregator()
        assert agg.aggregate([]) == []

    def test_two_buy_signals_same_symbol_both_pass(self):
        agg = SignalAggregator()
        s1 = _make_signal(strategy="momentum")
        s2 = _make_signal(strategy="mean_reversion")
        result = agg.aggregate([s1, s2])
        assert len(result) == 2
        assert s1 in result
        assert s2 in result

    def test_buy_sell_equal_ic_both_filtered(self):
        """BUY + SELL with equal IC -> net confidence = 0 -> both filtered."""
        agg = SignalAggregator()
        s1 = _make_signal(side=OrderSide.BUY, strategy="momentum")
        s2 = _make_signal(side=OrderSide.SELL, strategy="mean_reversion")
        result = agg.aggregate([s1, s2])
        assert len(result) == 0

    def test_buy_sell_different_ics_winner_passes(self):
        """BUY + SELL with different ICs -> higher-IC side wins."""
        agg = SignalAggregator(min_net_confidence=0.3)
        s_buy = _make_signal(side=OrderSide.BUY, strategy="momentum")
        s_sell = _make_signal(side=OrderSide.SELL, strategy="mean_reversion")
        ics = {"momentum": 0.8, "mean_reversion": 0.2}
        result = agg.aggregate([s_buy, s_sell], strategy_ics=ics)
        assert len(result) == 1
        assert result[0].side == OrderSide.BUY
        assert result[0].strategy == "momentum"

    def test_different_symbols_dont_interact(self):
        agg = SignalAggregator()
        s1 = _make_signal(symbol="AAPL", side=OrderSide.BUY, strategy="momentum")
        s2 = _make_signal(symbol="MSFT", side=OrderSide.SELL, strategy="mean_reversion")
        result = agg.aggregate([s1, s2])
        assert len(result) == 2
        symbols = {s.symbol for s in result}
        assert symbols == {"AAPL", "MSFT"}


class TestCorrelationPenalty:
    """Correlation penalty discounts redundant strategies."""

    def test_correlated_strategies_penalised(self):
        """Two perfectly correlated strategies -- lower-IC one gets 50% penalty."""
        agg = SignalAggregator(min_net_confidence=0.3, correlation_penalty_threshold=0.7)
        s_buy = _make_signal(side=OrderSide.BUY, strategy="strat_a")
        s_sell = _make_signal(side=OrderSide.SELL, strategy="strat_b")

        ics = {"strat_a": 0.6, "strat_b": 0.5}

        # Perfectly correlated histories
        history = {
            "strat_a": [1, 1, -1, 1, -1, 1, 1, -1],
            "strat_b": [1, 1, -1, 1, -1, 1, 1, -1],
        }

        result = agg.aggregate([s_buy, s_sell], strategy_ics=ics, recent_signal_history=history)

        # strat_b (lower IC=0.5) penalised to 0.25, strat_a stays at 0.6
        # net = 0.6 - 0.25 = 0.35, total = 0.85, confidence = 0.35/0.85 ~= 0.41 > 0.3
        # BUY wins
        assert len(result) == 1
        assert result[0].side == OrderSide.BUY

    def test_uncorrelated_strategies_no_penalty(self):
        """Uncorrelated strategies should not be penalised."""
        agg = SignalAggregator(min_net_confidence=0.3, correlation_penalty_threshold=0.7)
        s_buy = _make_signal(side=OrderSide.BUY, strategy="strat_a")
        s_sell = _make_signal(side=OrderSide.SELL, strategy="strat_b")

        ics = {"strat_a": 0.5, "strat_b": 0.5}

        # Low-correlation histories (Pearson ~0.14)
        history = {
            "strat_a": [1, 1, -1, 1, -1, 1, 1, -1],
            "strat_b": [1, -1, -1, 1, 1, -1, 1, -1],
        }

        result = agg.aggregate([s_buy, s_sell], strategy_ics=ics, recent_signal_history=history)
        # Equal ICs, no penalty -> net = 0 -> both filtered
        assert len(result) == 0

    def test_no_history_no_penalty(self):
        agg = SignalAggregator()
        s1 = _make_signal(strategy="a")
        s2 = _make_signal(strategy="b")
        result = agg.aggregate([s1, s2])
        assert len(result) == 2


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_three_buys_one_sell_same_symbol(self):
        """3 BUY vs 1 SELL with equal weights -> net = 2/4 = 0.5 > 0.3 -> BUYs win."""
        agg = SignalAggregator(min_net_confidence=0.3)
        signals = [
            _make_signal(side=OrderSide.BUY, strategy="s1"),
            _make_signal(side=OrderSide.BUY, strategy="s2"),
            _make_signal(side=OrderSide.BUY, strategy="s3"),
            _make_signal(side=OrderSide.SELL, strategy="s4"),
        ]
        result = agg.aggregate(signals)
        assert len(result) == 3
        assert all(s.side == OrderSide.BUY for s in result)

    def test_min_net_confidence_boundary(self):
        """Net confidence exactly at threshold should still be filtered."""
        # 2 BUY (w=1 each) + 1 SELL (w=1) -> net=1, total=3, conf=1/3 ~= 0.333
        agg = SignalAggregator(min_net_confidence=0.34)
        signals = [
            _make_signal(side=OrderSide.BUY, strategy="s1"),
            _make_signal(side=OrderSide.BUY, strategy="s2"),
            _make_signal(side=OrderSide.SELL, strategy="s3"),
        ]
        result = agg.aggregate(signals)
        # 1/3 = 0.333 < 0.34 -> filtered
        assert len(result) == 0
