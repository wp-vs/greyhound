"""Probability calibration for model outputs.

Raw model scores are not necessarily well-calibrated probabilities.
This module applies post-hoc calibration using:
- Platt scaling (sigmoid fit)
- Isotonic regression (non-parametric)

Well-calibrated probabilities are essential for:
1. Comparing model output to market odds (finding value)
2. Combining predictions from multiple models
3. Making Kelly criterion staking decisions
"""

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrate raw model probabilities to true probabilities.

    After calibration, a predicted probability of 0.20 should mean
    that the event occurs roughly 20% of the time.
    """

    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: "platt" (sigmoid/Platt scaling) or "isotonic"
                    (isotonic regression, more flexible but needs more data).
        """
        self.method = method
        self._calibrator = None
        self._fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ProbabilityCalibrator":
        """Fit the calibrator on validation data.

        Args:
            y_true: Binary outcomes (0/1).
            y_prob: Raw predicted probabilities.
        """
        if self.method == "platt":
            self._calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            self._calibrator.fit(y_prob.reshape(-1, 1), y_true)
        elif self.method == "isotonic":
            self._calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._calibrator.fit(y_prob, y_true)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._fitted = True
        logger.info(f"Calibrator ({self.method}) fitted on {len(y_true)} samples")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities.

        Args:
            y_prob: Raw predicted probabilities.

        Returns:
            Calibrated probabilities.
        """
        if not self._fitted or self._calibrator is None:
            raise RuntimeError("Calibrator not fitted")

        if self.method == "platt":
            return self._calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        else:
            return self._calibrator.predict(y_prob)

    @staticmethod
    def reliability_diagram(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Compute calibration curve data for a reliability diagram.

        Args:
            y_true: Binary outcomes.
            y_prob: Predicted probabilities.
            n_bins: Number of bins.

        Returns:
            DataFrame with mean_predicted, fraction_positive, and count columns.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        # Also get counts per bin
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        counts = np.bincount(bin_indices, minlength=n_bins)

        return pd.DataFrame({
            "mean_predicted": prob_pred,
            "fraction_positive": prob_true,
            "count": counts[:len(prob_pred)],
        })


def calibrate_within_race(probs: np.ndarray) -> np.ndarray:
    """Normalise probabilities within a race to sum to 1.

    This is a simple form of calibration that ensures
    the probabilities for all runners in a race are coherent.

    Args:
        probs: Array of probabilities for runners in a single race.

    Returns:
        Normalised probabilities summing to 1.
    """
    total = probs.sum()
    if total > 0:
        return probs / total
    return np.full_like(probs, 1.0 / len(probs))
