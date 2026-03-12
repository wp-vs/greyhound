"""Speed rating calculation for greyhound races.

Normalises raw finishing times to account for:
- Track variance (some tracks are faster/slower)
- Distance
- Going/track condition
- Sectional performance (early pace vs finishing speed)

Produces a single comparable "speed figure" per run.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class SpeedRatingCalculator:
    """Calculate normalised speed ratings from raw times.

    Methodology:
    1. Fit a regression: time ~ distance + track + going
    2. Speed rating = (predicted_time - actual_time) * scale_factor
       Positive = faster than expected, negative = slower.
    3. Optionally incorporate sectional times for pace analysis.
    """

    def __init__(self, scale_factor: float = 10.0):
        self.scale_factor = scale_factor
        self.model: LinearRegression | None = None
        self._track_means: dict[str, float] = {}
        self._distance_means: dict[int, float] = {}
        self._going_means: dict[str, float] = {}
        self._global_mean: float = 0.0
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "SpeedRatingCalculator":
        """Fit the speed rating model from historical data.

        Args:
            df: DataFrame with columns: finish_time, distance, track, going (optional).
        """
        data = df.dropna(subset=["finish_time", "distance"]).copy()
        if data.empty:
            logger.warning("No data with finish times to fit speed model")
            return self

        self._global_mean = data["finish_time"].mean()

        # Track means (as adjustment from global)
        self._track_means = (
            data.groupby("track")["finish_time"]
            .mean()
            .to_dict()
        )

        # Distance means
        self._distance_means = (
            data.groupby("distance")["finish_time"]
            .mean()
            .to_dict()
        )

        # Going means (if available)
        if "going" in data.columns and data["going"].notna().any():
            self._going_means = (
                data.groupby("going")["finish_time"]
                .mean()
                .to_dict()
            )

        # Fit regression model for more precise adjustment
        features = self._build_features(data)
        if features is not None and len(features) > 10:
            self.model = LinearRegression()
            self.model.fit(features, data.loc[features.index, "finish_time"])

        self._fitted = True
        logger.info(f"Speed rating model fitted on {len(data)} runs")
        return self

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate speed ratings for each run.

        Args:
            df: DataFrame with finish_time, distance, track columns.

        Returns:
            Series of speed ratings (higher = faster).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before calculate()")

        ratings = pd.Series(np.nan, index=df.index, dtype=float)
        mask = df["finish_time"].notna() & df["distance"].notna()
        data = df.loc[mask]

        if data.empty:
            return ratings

        if self.model is not None:
            features = self._build_features(data)
            if features is not None:
                predicted = self.model.predict(features)
                actual = data.loc[features.index, "finish_time"].values
                ratings.loc[features.index] = (predicted - actual) * self.scale_factor
                return ratings

        # Fallback: use track+distance means
        for idx, row in data.iterrows():
            expected = self._get_expected_time(row)
            if expected is not None:
                ratings.loc[idx] = (expected - row["finish_time"]) * self.scale_factor

        return ratings

    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame | None:
        """Build feature matrix for the regression model."""
        features = pd.DataFrame(index=data.index)

        # Distance as primary feature
        features["distance"] = data["distance"].values

        # Track dummies
        if "track" in data.columns:
            track_dummies = pd.get_dummies(data["track"], prefix="track", drop_first=True)
            features = features.join(track_dummies)

        # Going dummies
        if "going" in data.columns and data["going"].notna().any():
            going_dummies = pd.get_dummies(
                data["going"].fillna("unknown"), prefix="going", drop_first=True
            )
            features = features.join(going_dummies)

        if features.shape[1] < 1:
            return None

        return features.astype(float)

    def _get_expected_time(self, row: pd.Series) -> float | None:
        """Get expected time using precomputed means."""
        dist = row.get("distance")
        track = row.get("track")

        if dist in self._distance_means:
            base = self._distance_means[dist]
        else:
            return None

        # Adjust for track
        if track in self._track_means:
            track_adj = self._track_means[track] - self._global_mean
            base += track_adj

        return base


class SectionalAnalyser:
    """Analyse sectional (split) times for pace profiling.

    Greyhound races have distinct phases:
    1. Box to first bend (early pace / trap break)
    2. First bend to second bend (mid-race positioning)
    3. Run-in (finishing speed)

    Dogs that lead early but fade, or dogs that close late,
    have different profiles that suit different tracks/distances.
    """

    def calculate_pace_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pace metrics from sectional times.

        Args:
            df: DataFrame with sectional_time and finish_time columns.

        Returns:
            DataFrame with pace profile columns added.
        """
        df = df.copy()

        if "sectional_time" not in df.columns or "finish_time" not in df.columns:
            return df

        mask = df["sectional_time"].notna() & df["finish_time"].notna()

        # Early speed: sectional time relative to total
        df.loc[mask, "early_speed_pct"] = (
            df.loc[mask, "sectional_time"] / df.loc[mask, "finish_time"]
        )

        # Run-in time
        df.loc[mask, "run_in_time"] = (
            df.loc[mask, "finish_time"] - df.loc[mask, "sectional_time"]
        )

        # Classify pace style
        if mask.any():
            median_early = df.loc[mask, "early_speed_pct"].median()
            df.loc[mask, "pace_style"] = np.where(
                df.loc[mask, "early_speed_pct"] < median_early,
                "front_runner",
                "closer",
            )

        return df
