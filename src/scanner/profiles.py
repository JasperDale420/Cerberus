from abc import ABC, abstractmethod

from src.core.domain import Regime, SymbolFeatures


class ScannerProfile(ABC):
    """
    Base class for strategy-specific scanner profiles.
    """

    def filter(self, features: SymbolFeatures) -> bool:
        """
        Backwards-compatible alias for PRD terminology.
        """
        return self.min_requirements(features)

    @abstractmethod
    def min_requirements(self, features: SymbolFeatures) -> bool:
        """
        PRD: Hard prerequisites; if false, symbol not considered for this strategy.
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, features: SymbolFeatures, regime: Regime) -> float:
        """
        Returns a score (0.0 to 100.0) indicating how well the symbol fits the strategy.
        """
        raise NotImplementedError


class VWAPReversionProfile(ScannerProfile):
    """
    Scanner profile for VWAP Reversion strategy.
    """

    def __init__(
        self,
        min_price: float = 10.0,
        min_volume: float = 0.0,
        min_sigma: float = 2.0,
    ):
        self.min_price = min_price
        self.min_volume = min_volume
        self.min_sigma = min_sigma

    def min_requirements(self, features: SymbolFeatures) -> bool:
        # Basic liquidity checks
        if features.price < self.min_price:
            return False
        if features.avg_volume < self.min_volume:
            return False

        # Strategy specific checks (e.g. high volatility, or specific flow)
        # For now, just liquidity.
        return True

    def score(self, features: SymbolFeatures, regime: Regime) -> float:
        # Base score
        score = 50.0

        # Penalize if too far from VWAP? Or reward?
        # Reversion likes distance.
        score += abs(features.distance_from_vwap) * 1000  # e.g. 0.01 dist -> +10

        # Penalize low volatility
        score += features.atr_pct * 1000

        sigma = abs(features.price_zscore)
        score += (sigma - self.min_sigma) * 20.0

        return min(max(score, 0.0), 100.0)


class FlowMomentumProfile(ScannerProfile):
    """
    Scanner for Flow-Confirmed Momentum.
    Likes: Strong Option Flow Bias (Z-Score > 2 or < -2).
    """

    def __init__(self, min_flow_zscore: float = 2.5):
        self.min_flow_zscore = min_flow_zscore

    def min_requirements(self, features: SymbolFeatures) -> bool:
        # Check absolute flow score
        if abs(features.flow_zscore) < self.min_flow_zscore:
            return False
        return True

    def score(self, features: SymbolFeatures, regime: Regime) -> float:
        score = 60.0  # Base high because flow is rare/strong

        # Reward magnitude of flow
        extra_sigma = abs(features.flow_zscore) - self.min_flow_zscore
        score += extra_sigma * 10.0

        # Boost if aggressive flow share is high (if available)
        if features.aggressive_flow_share > 0.6:
            score += 10.0

        return min(max(score, 0.0), 100.0)


class GapProfile(ScannerProfile):
    """
    Scanner for Gap-Fill Candidates.
    Likes: Significant Morning Gaps (1.5% to 10%).
    Too small = noise. Too large = buyout/news (unfillable).
    """

    def __init__(self, min_gap: float = 0.015, max_gap: float = 0.10):
        self.min_gap = min_gap
        self.max_gap = max_gap

    def min_requirements(self, features: SymbolFeatures) -> bool:
        gap = abs(features.gap_pct)
        if gap < self.min_gap:
            return False
        if gap > self.max_gap:
            return False
        return True

    def score(self, features: SymbolFeatures, regime: Regime) -> float:
        # Score based on how "perfect" the gap is.
        # Maybe 3-5% is sweet spot?
        gap = abs(features.gap_pct)
        score = 50.0

        # Reward size
        score += (gap * 100.0) * 2.0

        return min(max(score, 0.0), 100.0)


class ORBScannerProfile(ScannerProfile):
    """
    Scanner for Opening Range Breakout candidates.
    Likes: Gaps, High Premarket Volume, High Relative Vol.
    """

    def __init__(self, min_gap_pct: float = 0.01, min_price: float = 10.0):
        self.min_gap_pct = min_gap_pct
        self.min_price = min_price

    def min_requirements(self, features: SymbolFeatures) -> bool:
        if features.price < self.min_price:
            return False

        # Must have decent gap OR high flow intensity
        has_gap = abs(features.gap_pct) >= self.min_gap_pct
        has_flow = abs(features.flow_zscore) > 2.0

        return has_gap or has_flow

    def score(self, features: SymbolFeatures, regime: Regime) -> float:
        score = 50.0

        # Reward Gap Size
        score += abs(features.gap_pct) * 500  # 1% gap -> +5

        # Reward Flow
        score += abs(features.flow_zscore) * 2.0

        # Reward Volatility
        score += features.atr_pct * 500

        return min(max(score, 0.0), 100.0)


class TrendPullbackProfile(ScannerProfile):
    """
    Scanner for Trend Pullback candidates.
    Likes: Strong Trend (ADX), Pullback to EMA20.
    """

    def __init__(self, min_adx: float = 25.0, max_dist_ema20: float = 0.02):
        self.min_adx = min_adx
        self.max_dist_ema20 = max_dist_ema20

    def min_requirements(self, features: SymbolFeatures) -> bool:
        # Check Trend Strength
        if features.adx < self.min_adx:
            return False

        # Check if Price is relatively close to EMA20 (Pullback zone)
        # Should not be too far away (extended)
        if abs(features.distance_from_ema20) > self.max_dist_ema20:
            return False

        return True

    def score(self, features: SymbolFeatures, regime: Regime) -> float:
        score = 50.0

        # Reward strong trend
        score += (features.adx - 25.0) * 1.0  # +1 per ADX point abovel 25

        # Reward proximity to EMA20 (closer is better for pullback entry, up to a point)
        # Ideally we want it sitting ON the EMA20.
        dist_score = (
            1.0 - (abs(features.distance_from_ema20) / self.max_dist_ema20)
        ) * 20.0
        score += max(0.0, dist_score)

        return min(max(score, 0.0), 100.0)


class FailedBreakoutProfile(ScannerProfile):
    """
    Scanner for Failed Breakout candidates.
    Likes: Price near Prior Day High/Low. High Volatility.
    """

    def __init__(self, proximity_pct: float = 0.02):
        self.proximity_pct = proximity_pct

    def min_requirements(self, features: SymbolFeatures) -> bool:
        pdh = features.prior_day_high
        pdl = features.prior_day_low
        price = features.price

        if pdh == 0.0 or pdl == 0.0:
            return False

        # Check proximity
        dist_high = abs(price - pdh) / pdh
        dist_low = abs(price - pdl) / pdl

        # We want to be watching it if it is near these levels
        return (dist_high < self.proximity_pct) or (dist_low < self.proximity_pct)

    def score(self, features: SymbolFeatures, regime: Regime) -> float:
        score = 50.0

        pdh = features.prior_day_high
        pdl = features.prior_day_low
        price = features.price

        if pdh == 0 or pdl == 0:
            return 0.0

        dist_high = abs(price - pdh) / pdh
        dist_low = abs(price - pdl) / pdl

        min_dist = min(dist_high, dist_low)

        # Reward closeness (closer = higher score)
        # 0% dist -> +20 score
        # 1% dist -> +10 score
        proximity_score = (self.proximity_pct - min_dist) / self.proximity_pct * 20.0
        score += max(0.0, proximity_score)

        # Reward Volatility
        score += features.atr_pct * 500

        return min(max(score, 0.0), 100.0)


class VWAPTrendRiderProfile(ScannerProfile):
    """
    Scanner for VWAP Trend Rider.
    Likes: Strong Trend (ADX > 20), Price near VWAP.
    """

    def __init__(self, min_adx: float = 20.0):
        self.min_adx = min_adx

    def min_requirements(self, features: SymbolFeatures) -> bool:
        if features.adx < self.min_adx:
            return False
        return True

    def score(self, features: SymbolFeatures, regime: Regime) -> float:
        score = 50.0
        score += (features.adx - 20.0) * 1.0

        # Reward proximity to VWAP
        dist = abs(features.distance_from_vwap)
        # Assuming we want it within 2%?
        dist_score = max(0.0, (0.02 - dist) / 0.02 * 20.0)
        score += dist_score

        return min(max(score, 0.0), 100.0)


class IndexMeanReversionProfile(ScannerProfile):
    """
    Scanner for Index Mean Reversion.
    Likes: Extreme Z-Score on Index ETFs.
    """

    def __init__(self, min_sigma: float = 2.0):
        self.min_sigma = min_sigma

    def min_requirements(self, features: SymbolFeatures) -> bool:
        if abs(features.price_zscore) < self.min_sigma:
            return False
        return True

    def score(self, features: SymbolFeatures, regime: Regime) -> float:
        score = 50.0 + (abs(features.price_zscore) * 10.0)
        return min(max(score, 0.0), 100.0)
