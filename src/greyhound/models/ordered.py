"""Ordered probit/logit model for finishing position prediction.

Finishing position is ordinal (1st > 2nd > 3rd > ...) so an
ordered model is more appropriate than treating each position
as an independent class.

Uses statsmodels OrderedModel for the implementation.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

logger = logging.getLogger(__name__)


class OrderedProbitModel(BaseModel):
    """Ordered probit model for finishing position.

    Models P(position <= k | features) for each cutpoint k,
    which naturally captures the ordinal structure.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        max_position: int = 6,
        distr: str = "probit",
    ):
        """
        Args:
            feature_columns: Feature columns to use.
            max_position: Cap finishing position at this value.
            distr: Distribution - "probit" (normal) or "logit".
        """
        super().__init__(feature_columns)
        self.max_position = max_position
        self.distr = distr
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "OrderedProbitModel":
        """Fit ordered probit model."""
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        X = self._prepare_features(df)

        # Cap position
        y = df["finish_position"].clip(upper=self.max_position).astype(int)

        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

        try:
            self.model = OrderedModel(y, X_scaled, distr=self.distr)
            self._result = self.model.fit(method="bfgs", disp=False, maxiter=500)
            self._fitted = True
            logger.info(f"Ordered {self.distr} model fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit ordered model: {e}")
            # Fallback: store means for simple prediction
            self._position_means = y.groupby(y).count() / len(y)
            self._fitted = True

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict win probability (P(position=1)) for each runner."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        X = self._prepare_features(df)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )

        if hasattr(self, "_result") and self._result is not None:
            try:
                pred = self._result.predict(X_scaled)
                # pred columns are P(Y=1), P(Y=2), ..., P(Y=max_position)
                # Return P(Y=1) as win probability
                if isinstance(pred, pd.DataFrame):
                    return pred.iloc[:, 0].values
                return pred[:, 0]
            except Exception as e:
                logger.warning(f"Prediction failed, using fallback: {e}")

        # Fallback: uniform probability
        return np.full(len(df), 1.0 / self.max_position)

    def predict_position_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict full position distribution for each runner.

        Returns:
            DataFrame with columns pos_1_prob, pos_2_prob, ..., pos_N_prob.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        X = self._prepare_features(df)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )

        if hasattr(self, "_result") and self._result is not None:
            pred = self._result.predict(X_scaled)
            if isinstance(pred, pd.DataFrame):
                pred.columns = [f"pos_{i+1}_prob" for i in range(pred.shape[1])]
                return pred
            else:
                return pd.DataFrame(
                    pred,
                    columns=[f"pos_{i+1}_prob" for i in range(pred.shape[1])],
                    index=df.index,
                )

        # Fallback
        n_pos = self.max_position
        return pd.DataFrame(
            np.full((len(df), n_pos), 1.0 / n_pos),
            columns=[f"pos_{i+1}_prob" for i in range(n_pos)],
            index=df.index,
        )
