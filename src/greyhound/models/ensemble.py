"""Ensemble model combining multiple prediction approaches.

Combines predictions from different model types to produce
more robust and accurate probability estimates. Uses either:
- Simple averaging (equal or weighted)
- Stacking (meta-learner on base model outputs)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble of multiple prediction models.

    Supports:
    1. Weighted average: simple weighted combination of probabilities
    2. Stacking: train a meta-model on base model predictions

    The ensemble typically outperforms any individual model because
    different models capture different aspects of the problem.
    """

    def __init__(
        self,
        models: list[BaseModel],
        weights: Optional[list[float]] = None,
        method: str = "weighted_average",
    ):
        """
        Args:
            models: List of fitted base models.
            weights: Weights for each model (must sum to 1).
                     None = equal weights.
            method: "weighted_average" or "stacking".
        """
        super().__init__()
        self.models = models
        self.method = method

        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("weights must match number of models")
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(models)] * len(models)

        self.meta_model: LogisticRegression | None = None

    def fit(self, df: pd.DataFrame) -> "EnsembleModel":
        """Fit the ensemble.

        For weighted_average: no additional fitting needed (models already fitted).
        For stacking: train a meta-learner on base model predictions.
        """
        if self.method == "stacking":
            self._fit_stacking(df)

        self._fitted = True
        return self

    def _fit_stacking(self, df: pd.DataFrame):
        """Train meta-learner on base model outputs."""
        # Get predictions from each base model
        meta_features = self._get_meta_features(df)

        y = df["won"].values

        self.meta_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
        )
        self.meta_model.fit(meta_features, y)

        logger.info(
            f"Stacking meta-model fitted. "
            f"Weights: {dict(zip([m.name for m in self.models], self.meta_model.coef_[0]))}"
        )

    def _get_meta_features(self, df: pd.DataFrame) -> np.ndarray:
        """Get predictions from all base models as meta-features."""
        predictions = []
        for model in self.models:
            try:
                probs = model.predict_proba(df)
                predictions.append(probs)
            except Exception as e:
                logger.warning(f"Model {model.name} prediction failed: {e}")
                predictions.append(np.full(len(df), 1.0 / 6.0))

        return np.column_stack(predictions)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict win probability using the ensemble."""
        if self.method == "stacking" and self.meta_model is not None:
            meta_features = self._get_meta_features(df)
            return self.meta_model.predict_proba(meta_features)[:, 1]

        # Weighted average
        predictions = self._get_meta_features(df)
        weights = np.array(self.weights)
        return predictions @ weights

    def optimize_weights(
        self, val_df: pd.DataFrame, metric: str = "log_loss"
    ) -> list[float]:
        """Optimize ensemble weights on validation data.

        Uses grid search over weight combinations to find the
        weights that minimize the chosen metric.

        Args:
            val_df: Validation DataFrame.
            metric: "log_loss" or "brier" score to minimize.

        Returns:
            Optimised weights.
        """
        from sklearn.metrics import brier_score_loss, log_loss

        predictions = self._get_meta_features(val_df)
        y = val_df["won"].values

        best_score = float("inf")
        best_weights = self.weights

        n_models = len(self.models)
        # Grid search with 0.05 step size
        steps = 20

        def _generate_weights(n, total_steps):
            """Generate all weight combinations that sum to 1."""
            if n == 1:
                yield [1.0]
                return
            for i in range(total_steps + 1):
                w = i / total_steps
                for rest in _generate_weights(n - 1, total_steps - i):
                    yield [w] + [r * (1 - w) for r in rest]

        for weights in _generate_weights(n_models, steps):
            weights_arr = np.array(weights)
            if abs(sum(weights) - 1.0) > 0.01:
                continue

            combined = predictions @ weights_arr
            combined = np.clip(combined, 1e-7, 1 - 1e-7)

            if metric == "log_loss":
                score = log_loss(y, combined)
            else:
                score = brier_score_loss(y, combined)

            if score < best_score:
                best_score = score
                best_weights = list(weights)

        self.weights = best_weights
        logger.info(
            f"Optimised weights: {dict(zip([m.name for m in self.models], best_weights))}"
        )
        logger.info(f"Best {metric}: {best_score:.6f}")

        return best_weights
