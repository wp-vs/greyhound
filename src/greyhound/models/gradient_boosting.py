"""Gradient boosting models (XGBoost, LightGBM) for race prediction.

These are the workhorse models in practice. They handle:
- Non-linear feature interactions (trap x track, form x grade)
- Missing values natively
- Feature importance ranking
- High predictive performance
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from .base import BaseModel

logger = logging.getLogger(__name__)


class GradientBoostingModel(BaseModel):
    """Gradient boosting model using XGBoost or LightGBM.

    Supports both binary classification (win/no-win) and
    ranking (LambdaRank for ordering runners within a race).
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        engine: Literal["xgboost", "lightgbm"] = "xgboost",
        objective: Literal["binary", "rank"] = "binary",
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        early_stopping_rounds: int = 50,
    ):
        super().__init__(feature_columns)
        self.engine = engine
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self._actual_features: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> "GradientBoostingModel":
        """Fit gradient boosting model.

        Args:
            df: Training data.
            val_df: Optional validation data for early stopping.
        """
        X_train = self._prepare_features(df)
        y_train = df["won"].values
        self._actual_features = list(X_train.columns)

        eval_set = None
        if val_df is not None:
            X_val = self._prepare_features(val_df)
            y_val = val_df["won"].values
            eval_set = [(X_val, y_val)]

        if self.engine == "xgboost":
            self._fit_xgboost(X_train, y_train, eval_set)
        else:
            self._fit_lightgbm(X_train, y_train, eval_set)

        self._fitted = True
        return self

    def _fit_xgboost(self, X, y, eval_set):
        import xgboost as xgb

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
            "n_jobs": -1,
        }

        fit_params = {}
        if eval_set:
            fit_params["eval_set"] = eval_set

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y, **fit_params)
        logger.info(f"XGBoost fitted with {self.model.best_iteration} trees")

    def _fit_lightgbm(self, X, y, eval_set):
        import lightgbm as lgb

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_samples": self.min_child_weight,
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "n_jobs": -1,
        }

        callbacks = []
        if eval_set:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        logger.info("LightGBM fitted successfully")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict win probability for each runner."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted")

        X = self._prepare_features(df)
        # Ensure same columns as training
        for col in self._actual_features:
            if col not in X.columns:
                X[col] = 0
        X = X[self._actual_features]

        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings."""
        if self.model is None:
            return pd.DataFrame()

        if self.engine == "xgboost":
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_

        return pd.DataFrame({
            "feature": self._actual_features[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False)


class RankingModel(BaseModel):
    """LambdaRank model for ordering runners within a race.

    Instead of predicting win/no-win, this directly optimises
    the ranking of runners within each race using pairwise
    learning-to-rank.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
    ):
        super().__init__(feature_columns)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self._actual_features: list[str] = []

    def fit(self, df: pd.DataFrame) -> "RankingModel":
        """Fit LambdaRank model using XGBoost.

        The target is inverted finish position (higher = better)
        and groups are defined by race_id.
        """
        import xgboost as xgb

        X = self._prepare_features(df)
        self._actual_features = list(X.columns)

        # Target: invert position so winner has highest value
        max_pos = df["finish_position"].max()
        y = max_pos + 1 - df["finish_position"].values

        # Group sizes (number of runners per race)
        if "race_id" in df.columns:
            groups = df.groupby("race_id", sort=False).size().values
        else:
            groups = np.array([6] * (len(df) // 6))  # assume 6-runner races

        self.model = xgb.XGBRanker(
            objective="rank:pairwise",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbosity=0,
        )
        self.model.fit(X, y, group=groups)
        self._fitted = True
        logger.info("LambdaRank model fitted")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict scores and convert to probabilities via softmax."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted")

        X = self._prepare_features(df)
        for col in self._actual_features:
            if col not in X.columns:
                X[col] = 0
        X = X[self._actual_features]

        scores = self.model.predict(X)

        # Convert scores to probabilities within each race
        if "race_id" in df.columns:
            probs = np.zeros_like(scores)
            for race_id in df["race_id"].unique():
                mask = df["race_id"] == race_id
                race_scores = scores[mask.values]
                # Softmax
                exp_scores = np.exp(race_scores - race_scores.max())
                probs[mask.values] = exp_scores / exp_scores.sum()
            return probs
        else:
            # Single race softmax
            exp_scores = np.exp(scores - scores.max())
            return exp_scores / exp_scores.sum()
