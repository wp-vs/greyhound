"""Class/grade adjustment modelling.

Greyhounds move up and down through race grades. Performance at
different grades is not directly comparable — a 1st place in
A5 is different from a 1st place in A1.

This module adjusts performance metrics to account for the
quality of competition faced.
"""

import logging

import numpy as np
import pandas as pd

from ..data.schema import GRADE_NUMERIC

logger = logging.getLogger(__name__)


class ClassAdjuster:
    """Adjust performance metrics for race grade/class.

    Methodology:
    1. Estimate the average winning time at each grade+distance+track combo.
    2. The difference between grades gives a "class length" — how many
       seconds separating each grade level.
    3. Adjust raw times to a common baseline grade.
    """

    def __init__(self, baseline_grade: str = "A5"):
        self.baseline_grade = baseline_grade
        self._adjustments: dict[tuple[str, int, str], float] = {}
        self._grade_time_diffs: pd.DataFrame | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "ClassAdjuster":
        """Fit class adjustments from historical data.

        Args:
            df: DataFrame with grade, distance, track, finish_time, finish_position columns.
        """
        data = df.dropna(subset=["grade", "distance", "finish_time"]).copy()
        if data.empty:
            logger.warning("No data to fit class adjustment model")
            return self

        data["grade_numeric"] = data["grade"].map(GRADE_NUMERIC)
        data = data.dropna(subset=["grade_numeric"])

        # Average winning time by grade + distance + track
        winners = data[data["finish_position"] == 1]
        mean_times = (
            winners.groupby(["track", "distance", "grade"])["finish_time"]
            .mean()
            .reset_index()
            .rename(columns={"finish_time": "mean_winning_time"})
        )

        # For each track+distance, compute adjustment per grade
        # relative to baseline
        baseline_num = GRADE_NUMERIC.get(self.baseline_grade, 10)

        for (track, dist), group in mean_times.groupby(["track", "distance"]):
            group = group.copy()
            group["grade_numeric"] = group["grade"].map(GRADE_NUMERIC)
            group = group.dropna(subset=["grade_numeric"]).sort_values("grade_numeric")

            if len(group) < 2:
                continue

            # Fit linear relationship: time = f(grade_numeric)
            coeffs = np.polyfit(group["grade_numeric"], group["mean_winning_time"], 1)
            # coeffs[0] = time per grade level
            time_per_grade = coeffs[0]

            for _, row in group.iterrows():
                grade_num = row["grade_numeric"]
                # Adjustment to bring to baseline grade
                adj = (grade_num - baseline_num) * time_per_grade
                self._adjustments[(track, dist, row["grade"])] = adj

        self._fitted = True
        logger.info(
            f"Class adjustment fitted: {len(self._adjustments)} track/dist/grade combos"
        )
        return self

    def adjust_time(
        self, finish_time: float, track: str, distance: int, grade: str
    ) -> float:
        """Adjust a finishing time to the baseline grade.

        Args:
            finish_time: Raw finishing time in seconds.
            track: Track name.
            distance: Race distance in metres.
            grade: Race grade.

        Returns:
            Adjusted time (as if run at baseline grade).
        """
        key = (track, distance, grade)
        adj = self._adjustments.get(key, 0.0)
        return finish_time - adj

    def add_adjusted_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add class-adjusted time column to DataFrame."""
        df = df.copy()

        def _adjust(row):
            if pd.isna(row.get("finish_time")) or pd.isna(row.get("grade")):
                return np.nan
            return self.adjust_time(
                row["finish_time"],
                row.get("track", ""),
                row.get("distance", 0),
                row["grade"],
            )

        df["adjusted_time"] = df.apply(_adjust, axis=1)
        return df
