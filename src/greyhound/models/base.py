"""Base model interface for greyhound race prediction."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


# Default feature columns used by models
FEATURE_COLUMNS = [
    # Form features
    "form_weighted_pos",
    "form_weighted_time",
    "form_weighted_win_rate",
    "form_trajectory",
    "form_consistency",
    "form_days_off",
    "form_run_count",
    # Career stats
    "career_win_rate",
    "career_place_rate",
    "career_runs",
    # Recent performance
    "avg_position_last5",
    "avg_position_last3",
    "best_position_last5",
    "avg_time_last5",
    "best_time_last5",
    "time_std_last5",
    # Speed rating
    "speed_rating",
    # Track/trap features
    "trap",
    "trap_bias",
    "runs_at_track",
    "runs_at_distance",
    # Class features
    "grade_numeric",
    "grade_change",
    "avg_grade_last3",
    "adjusted_time",
    # Elo
    "elo_rating",
    # Market
    "sp_implied_prob",
    # Race context
    "distance",
    "days_since_last",
]


class BaseModel(ABC):
    """Abstract base for prediction models."""

    def __init__(self, feature_columns: list[str] | None = None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self._fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseModel":
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict win probabilities for each runner.

        Args:
            df: DataFrame with one row per runner per race.

        Returns:
            Array of win probabilities, same length as df.
        """
        ...

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities for a single race.

        Args:
            race_df: DataFrame with one row per runner in the race.

        Returns:
            DataFrame with predicted_win_prob and predicted_rank columns.
        """
        probs = self.predict_proba(race_df)
        result = race_df.copy()
        result["predicted_win_prob"] = probs

        # Normalise within race so probabilities sum to 1
        total = result["predicted_win_prob"].sum()
        if total > 0:
            result["predicted_win_prob"] = result["predicted_win_prob"] / total

        result["predicted_rank"] = result["predicted_win_prob"].rank(
            ascending=False, method="min"
        )
        return result

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and clean feature matrix."""
        available = [c for c in self.feature_columns if c in df.columns]
        X = df[available].copy()
        X = X.fillna(X.median())
        # Fill any remaining NaN with 0 (for columns that were all NaN)
        X = X.fillna(0)
        return X

    @property
    def name(self) -> str:
        return self.__class__.__name__
