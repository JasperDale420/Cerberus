"""Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

Normal-Inverse-Gamma conjugate prior with constant hazard function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import t as student_t

from src.core.logger import StructuredLogger

# Run lengths below this threshold contribute to "recent changepoint" probability.
_CP_RUN_LENGTH_THRESHOLD = 5


@dataclass
class BOCPDResult:
    """Posterior summary after one BOCPD update."""

    changepoint_probability: float  # P(r_t < threshold), probability a changepoint occurred recently
    map_run_length: int  # argmax of posterior
    expected_run_length: float  # E[r_t]
    growth_probabilities: np.ndarray  # full posterior over run lengths


class BOCPDDetector:
    """Online Bayesian changepoint detector with NIG conjugate prior."""

    def __init__(
        self,
        hazard_lambda: float = 200,
        prior_mu: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 0.01,
        max_run_length: int = 200,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self._hazard = 1.0 / hazard_lambda
        self._prior_mu = prior_mu
        self._prior_kappa = prior_kappa
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._max_run_length = max_run_length
        self._logger = logger

        self._t = 0
        self._mu = np.full(max_run_length, prior_mu)
        self._kappa = np.full(max_run_length, prior_kappa)
        self._alpha = np.full(max_run_length, prior_alpha)
        self._beta = np.full(max_run_length, prior_beta)

        # Joint probability P(r_t, x_{1:t})
        self._joint = np.zeros(max_run_length)
        self._joint[0] = 1.0

    def update(self, value: float) -> BOCPDResult:
        """Feed one observation and return the posterior over run lengths."""
        r_max = min(self._t + 1, self._max_run_length)

        # --- Predictive probabilities under each run length ---
        pred_probs = self._predictive_probs(value, r_max)

        # --- Message passing ---
        new_joint = np.zeros(self._max_run_length)
        weighted = pred_probs * self._joint[:r_max]

        # Growth: P(r_{t+1}=r+1) = pred[r] * joint[r] * (1-H)
        grow_end = min(r_max + 1, self._max_run_length)
        new_joint[1:grow_end] = weighted[: grow_end - 1] * (1.0 - self._hazard)

        # Changepoint: P(r_{t+1}=0) = sum(pred[r] * joint[r] * H)
        new_joint[0] = np.sum(weighted) * self._hazard

        # Normalize
        evidence = new_joint.sum()
        if evidence > 1e-300:
            new_joint /= evidence
        else:
            # Numerical underflow — reset to changepoint prior to recover
            new_joint = np.zeros(self._max_run_length)
            new_joint[0] = 1.0
        self._joint = new_joint

        # --- Update sufficient statistics ---
        self._update_sufficient_stats(value, r_max)
        self._t += 1

        # --- Build result ---
        posterior = self._joint.copy()
        map_rl = int(np.argmax(posterior))
        expected_rl = float(np.dot(np.arange(len(posterior)), posterior))

        # Changepoint probability: mass concentrated at short run lengths
        # After a changepoint, the posterior shifts to small r values.
        cp_threshold = min(_CP_RUN_LENGTH_THRESHOLD, self._max_run_length)
        cp_prob = float(np.sum(posterior[:cp_threshold]))

        if cp_prob > 0.5 and self._logger:
            self._logger.info(
                "bocpd_changepoint_detected",
                t=self._t,
                changepoint_probability=round(cp_prob, 4),
                map_run_length=map_rl,
            )

        return BOCPDResult(
            changepoint_probability=cp_prob,
            map_run_length=map_rl,
            expected_run_length=expected_rl,
            growth_probabilities=posterior,
        )

    def reset(self) -> None:
        """Reset all internal state to initial conditions."""
        self._t = 0
        self._mu = np.full(self._max_run_length, self._prior_mu)
        self._kappa = np.full(self._max_run_length, self._prior_kappa)
        self._alpha = np.full(self._max_run_length, self._prior_alpha)
        self._beta = np.full(self._max_run_length, self._prior_beta)
        self._joint = np.zeros(self._max_run_length)
        self._joint[0] = 1.0

    def _predictive_probs(self, value: float, r_max: int) -> np.ndarray:
        """Compute Student-t predictive probability for each run length."""
        df = 2.0 * self._alpha[:r_max]
        loc = self._mu[:r_max]
        kappa_slice = self._kappa[:r_max]
        beta_slice = self._beta[:r_max]
        alpha_slice = self._alpha[:r_max]

        scale = np.sqrt(beta_slice * (kappa_slice + 1.0) / (alpha_slice * kappa_slice))
        scale = np.maximum(scale, 1e-300)
        df = np.maximum(df, 1e-300)

        pred = student_t.pdf(value, df=df, loc=loc, scale=scale)
        return np.maximum(pred, 1e-300)

    def _update_sufficient_stats(self, value: float, r_max: int) -> None:
        """Update NIG sufficient statistics for all run lengths."""
        new_mu = np.full(self._max_run_length, self._prior_mu)
        new_kappa = np.full(self._max_run_length, self._prior_kappa)
        new_alpha = np.full(self._max_run_length, self._prior_alpha)
        new_beta = np.full(self._max_run_length, self._prior_beta)

        update_len = min(r_max, self._max_run_length - 1)
        old_mu = self._mu[:update_len]
        old_kappa = self._kappa[:update_len]
        old_alpha = self._alpha[:update_len]
        old_beta = self._beta[:update_len]

        new_kappa_vals = old_kappa + 1.0
        new_mu[1 : update_len + 1] = (old_kappa * old_mu + value) / new_kappa_vals
        new_kappa[1 : update_len + 1] = new_kappa_vals
        new_alpha[1 : update_len + 1] = old_alpha + 0.5
        new_beta[1 : update_len + 1] = old_beta + 0.5 * old_kappa * (value - old_mu) ** 2 / new_kappa_vals

        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta
