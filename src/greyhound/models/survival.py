"""Survival / competing risks model for race prediction.

Frames a greyhound race as a competing risks problem:
each runner "fails" (is beaten) at different points. The Cox
proportional hazards model estimates how features affect the
hazard of being beaten.

This is an elegant framework for racing because:
- It naturally handles the ordered nature of finishing positions
- It can incorporate censored observations (non-finishers)
- The partial likelihood doesn't require specifying the baseline hazard
"""

import logging

import numpy as np
import pandas as pd

from .base import BaseModel

logger = logging.getLogger(__name__)


class SurvivalModel(BaseModel):
    """Cox proportional hazards model for race outcomes.

    Each runner's finishing position is treated as a "time to event",
    where the event is being beaten. The winner has the longest
    "survival time" (position 1 = last to be beaten).
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        penalizer: float = 0.1,
    ):
        super().__init__(feature_columns)
        self.penalizer = penalizer
        self.model = None

    def fit(self, df: pd.DataFrame) -> "SurvivalModel":
        """Fit Cox PH model on training data.

        Args:
            df: Training data with finish_position and feature columns.
        """
        from lifelines import CoxPHFitter

        X = self._prepare_features(df)

        # Duration = inverted position (winner=6, last=1 in 6-runner race)
        max_pos = df["finish_position"].max()
        X["duration"] = max_pos + 1 - df["finish_position"].values
        X["event"] = 1  # all runners complete (observed)

        self.model = CoxPHFitter(penalizer=self.penalizer)

        try:
            self.model.fit(
                X,
                duration_col="duration",
                event_col="event",
                show_progress=False,
            )
            self._fitted = True
            logger.info("Cox PH model fitted successfully")

            # Log top features
            summary = self.model.summary
            top = summary.nlargest(10, "coef")
            logger.info("Top Cox PH features:")
            for feat, row in top.iterrows():
                logger.info(f"  {feat}: coef={row['coef']:.4f}, p={row['p']:.4f}")

        except Exception as e:
            logger.error(f"Cox PH fitting failed: {e}")
            self._fitted = True  # still mark as fitted, will use fallback

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict win probability using partial hazards.

        Higher partial hazard = more likely to "survive" longer
        = more likely to finish in a better position.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        X = self._prepare_features(df)

        if self.model is not None and hasattr(self.model, "predict_partial_hazard"):
            try:
                # Partial hazard: exp(X @ beta)
                hazards = self.model.predict_partial_hazard(X).values.flatten()

                # Convert to probabilities within each race
                if "race_id" in df.columns:
                    probs = np.zeros_like(hazards)
                    for race_id in df["race_id"].unique():
                        mask = (df["race_id"] == race_id).values
                        race_h = hazards[mask]
                        probs[mask] = race_h / race_h.sum()
                    return probs
                else:
                    return hazards / hazards.sum()

            except Exception as e:
                logger.warning(f"Cox prediction failed: {e}")

        # Fallback: uniform
        return np.full(len(df), 1.0 / 6.0)

    def feature_importance(self) -> pd.DataFrame:
        """Get Cox model coefficients."""
        if self.model is None:
            return pd.DataFrame()

        summary = self.model.summary.copy()
        summary["abs_coef"] = summary["coef"].abs()
        return summary.sort_values("abs_coef", ascending=False)
