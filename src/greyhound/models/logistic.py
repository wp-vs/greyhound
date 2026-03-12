"""Logistic regression model for win/place prediction.

The simplest parametric approach: model P(win | features)
using logistic regression. Interpretable coefficients show
which features matter most.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

logger = logging.getLogger(__name__)


class LogisticModel(BaseModel):
    """Logistic regression for binary win prediction."""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        super().__init__(feature_columns)
        self.C = C
        self.max_iter = max_iter
        self.model: LogisticRegression | None = None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "LogisticModel":
        """Fit logistic regression on training data.

        Args:
            df: Training DataFrame with feature columns and 'won' target.
        """
        X = self._prepare_features(df)
        y = df["won"].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight="balanced",
            solver="lbfgs",
        )
        self.model.fit(X_scaled, y)
        self._fitted = True

        # Log feature importances
        coefs = dict(zip(X.columns, self.model.coef_[0]))
        top_features = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        logger.info("Top logistic regression features:")
        for feat, coef in top_features:
            logger.info(f"  {feat}: {coef:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict win probability for each runner."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted")

        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        """Get feature coefficients."""
        if self.model is None:
            return pd.DataFrame()

        available = [c for c in self.feature_columns]
        return pd.DataFrame({
            "feature": available[:len(self.model.coef_[0])],
            "coefficient": self.model.coef_[0],
            "abs_coefficient": np.abs(self.model.coef_[0]),
        }).sort_values("abs_coefficient", ascending=False)


class PlaceModel(LogisticModel):
    """Logistic regression for place (top 3) prediction."""

    def fit(self, df: pd.DataFrame) -> "PlaceModel":
        X = self._prepare_features(df)
        y = df["placed"].values

        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight="balanced",
            solver="lbfgs",
        )
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self
