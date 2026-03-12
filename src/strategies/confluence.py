"""
Confluence Scoring — weighted multi-factor signal evaluation.

Replaces binary "all conditions met" logic with probabilistic entry.
Each factor is scored 0-100 with a weight 0.0-1.0; the weighted average
produces a confluence score.  Only signals above a configurable threshold
are acted on, and the score maps to a conviction multiplier for position
sizing.

Usage::

    from cerberus.strategies.confluence import ConfluenceScorer, score_deviation

    scorer = ConfluenceScorer(threshold=65.0)
    scorer.add_factor("vwap_deviation", raw_value=-2.1, score=85, weight=0.25)
    scorer.add_factor("rsi_extremity",  raw_value=8.5,  score=90, weight=0.20)
    scorer.add_factor("volume_spike",   raw_value=2.3,  score=70, weight=0.15)
    scorer.add_factor("reversal_candle", raw_value=1.8, score=60, weight=0.20)
    scorer.add_factor("flow_alignment", raw_value=0.3,  score=40, weight=0.20)

    if scorer.passes_threshold():
        signal.meta.update(scorer.to_meta())
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class ConfluenceFactor:
    """A single scored factor in the confluence assessment."""

    name: str
    raw_value: float  # The actual indicator value (for logging)
    score: float  # Normalized 0-100 score
    weight: float  # 0.0-1.0 weight in final score
    passed: bool  # Whether this factor individually "passed" its threshold


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ConfluenceScorer:
    """
    Weighted multi-factor confluence scoring for trade entry decisions.

    Factors are added one-by-one via :meth:`add_factor`, then the weighted
    average is compared against *threshold*.  A conviction multiplier maps
    the score to a 0.5-1.0 range for position-size scaling.
    """

    def __init__(self, threshold: float = 65.0) -> None:
        self.threshold = threshold
        self.factors: list[ConfluenceFactor] = []

    # -- mutators -----------------------------------------------------------

    def add_factor(
        self,
        name: str,
        raw_value: float,
        score: float,
        weight: float,
        passed: bool = True,
    ) -> None:
        """Add a scored factor.  *score* should be 0-100, *weight* 0.0-1.0."""
        clamped_score = max(0.0, min(100.0, float(score)))
        clamped_weight = max(0.0, min(1.0, float(weight)))
        self.factors.append(
            ConfluenceFactor(
                name=name,
                raw_value=float(raw_value),
                score=clamped_score,
                weight=clamped_weight,
                passed=passed,
            )
        )

    def reset(self) -> None:
        """Clear all factors for reuse."""
        self.factors.clear()

    # -- queries ------------------------------------------------------------

    def score(self) -> float:
        """Weighted average score (0-100).

        Formula::

            sum(score_i * weight_i) / sum(weight_i)

        Returns 0.0 when no factors have been added.
        """
        total_weight = sum(f.weight for f in self.factors)
        if abs(total_weight) < 1e-9:
            return 0.0
        return sum(f.score * f.weight for f in self.factors) / total_weight

    def passes_threshold(self) -> bool:
        """``True`` if the weighted score >= *threshold*."""
        return self.score() >= self.threshold

    def conviction_multiplier(self) -> float:
        """Position-size multiplier based on conviction.

        Maps the confluence score to 0.5-1.0:

        * Score at *threshold* (e.g. 65)  -> 0.5  (minimum conviction)
        * Score at 100                     -> 1.0  (maximum conviction)
        * Below threshold                  -> 0.0  (no trade)
        """
        s = self.score()
        if s < self.threshold:
            return 0.0
        spread = 100.0 - self.threshold
        if spread <= 0.0:
            return 1.0
        ratio = (s - self.threshold) / spread
        return 0.5 + 0.5 * ratio

    def to_meta(self) -> dict:
        """Export scoring details as a dict suitable for ``Signal.meta``.

        Returns::

            {
                "confluence_score": 78.5,
                "confluence_threshold": 65.0,
                "conviction_multiplier": 0.69,
                "factors": {
                    "vwap_deviation": {"raw": -2.1, "score": 85, "weight": 0.25, "passed": True},
                    ...
                }
            }
        """
        return {
            "confluence_score": round(self.score(), 2),
            "confluence_threshold": self.threshold,
            "conviction_multiplier": round(self.conviction_multiplier(), 4),
            "factors": {
                f.name: {
                    "raw": f.raw_value,
                    "score": f.score,
                    "weight": f.weight,
                    "passed": f.passed,
                }
                for f in self.factors
            },
        }


# ---------------------------------------------------------------------------
# Scoring helper functions
# ---------------------------------------------------------------------------


def score_deviation(value: float, min_threshold: float, max_value: float) -> float:
    """Score a deviation on 0-100 (linear between *min_threshold* and *max_value*).

    Works on absolute values (distance from zero).

    Examples::

        score_deviation(abs(-2.1), 1.0, 3.0)  # -> 55.0
        score_deviation(0.5, 1.0, 3.0)         # -> 0.0
        score_deviation(4.0, 1.0, 3.0)         # -> 100.0
    """
    v = abs(value)
    if v <= min_threshold:
        return 0.0
    if v >= max_value:
        return 100.0
    return (v - min_threshold) / (max_value - min_threshold) * 100.0


def score_threshold(
    value: float,
    threshold: float,
    max_value: float,
    invert: bool = False,
) -> float:
    """Score a value that must exceed (or be below, if *invert*) a threshold.

    Non-inverted (e.g. volume > 1.5x)::

        Below threshold -> 0
        At threshold    -> 50
        At max_value    -> 100

    Inverted (e.g. RSI < 10)::

        Above threshold -> 0
        At threshold    -> 50
        At 0            -> 100
    """
    if not invert:
        if value < threshold:
            return 0.0
        if value >= max_value:
            return 100.0
        spread = max_value - threshold
        if spread <= 0.0:
            return 100.0
        return 50.0 + 50.0 * (value - threshold) / spread
    else:
        if value > threshold:
            return 0.0
        if value <= 0.0:
            return 100.0
        if threshold <= 0.0:
            return 100.0
        return 50.0 + 50.0 * (threshold - value) / threshold


def score_candle_quality(
    open_price: float,
    high: float,
    low: float,
    close: float,
    side: str,
) -> float:
    """Score a candle's reversal quality (0-100) for a given *side*.

    Factors (buy = bullish reversal / sell = bearish reversal):

    * Wick ratio  (desired wick / body): 0-40 pts
    * Close position in range:           0-30 pts
    * Body direction (correct sign):     0-30 pts
    """
    candle_range = high - low
    if candle_range <= 0.0:
        return 0.0

    body = abs(close - open_price)
    body = max(body, candle_range * 0.001)  # avoid div-zero

    is_buy = side.lower() in ("buy", "long")

    # -- wick ratio (0-40) --------------------------------------------------
    if is_buy:
        desired_wick = open_price - low if close >= open_price else close - low
    else:
        desired_wick = high - open_price if close <= open_price else high - close
    desired_wick = max(desired_wick, 0.0)
    wick_ratio = desired_wick / body
    wick_score = min(wick_ratio / 2.0, 1.0) * 40.0  # ratio of 2 -> full marks

    # -- close position (0-30) ----------------------------------------------
    close_pct = (close - low) / candle_range  # 0 = at low, 1 = at high
    if is_buy:
        close_score = close_pct * 30.0
    else:
        close_score = (1.0 - close_pct) * 30.0

    # -- body direction (0-30) ----------------------------------------------
    if is_buy:
        direction_score = 30.0 if close > open_price else 0.0
    else:
        direction_score = 30.0 if close < open_price else 0.0

    return min(wick_score + close_score + direction_score, 100.0)


def score_volume(
    current_volume: float,
    average_volume: float,
    min_mult: float = 1.0,
    ideal_mult: float = 2.5,
) -> float:
    """Score volume relative to average (0-100).

    * Below *min_mult* x average -> 0
    * At *min_mult*              -> 30
    * At *ideal_mult*            -> 100

    Example::

        score_volume(350_000, 200_000, 1.0, 2.5)  # 1.75x -> ~63.3
    """
    if average_volume <= 0.0:
        return 0.0
    mult = current_volume / average_volume
    if mult < min_mult:
        return 0.0
    if mult >= ideal_mult:
        return 100.0
    spread = ideal_mult - min_mult
    if spread <= 0.0:
        return 100.0
    return 30.0 + 70.0 * (mult - min_mult) / spread
