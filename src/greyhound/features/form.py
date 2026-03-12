"""Form sequencing and recency-weighted performance modelling.

Models the trajectory and quality of a greyhound's recent runs,
applying time-decay weighting so more recent performances count more.
"""

import numpy as np
import pandas as pd


class FormCalculator:
    """Calculate form-based features with time-decay weighting.

    Recent runs are weighted exponentially more than older runs.
    A half-life parameter controls how quickly older form decays.
    """

    def __init__(self, half_life_days: float = 28.0, max_lookback: int = 10):
        """
        Args:
            half_life_days: Days after which a run's weight halves.
            max_lookback: Maximum number of prior runs to consider.
        """
        self.half_life_days = half_life_days
        self.max_lookback = max_lookback

    def calculate_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-decay weighted form features.

        Args:
            df: DataFrame sorted by date with greyhound, date, finish_position,
                finish_time, distance, grade_numeric columns.

        Returns:
            DataFrame with form features added per row.
        """
        df = df.copy()
        df = df.sort_values(["greyhound", "date"]).reset_index(drop=True)

        form_features = []

        for _, group in df.groupby("greyhound"):
            group = group.sort_values("date")
            feats = self._compute_form_for_dog(group)
            form_features.append(feats)

        if not form_features:
            return df

        form_df = pd.concat(form_features, ignore_index=True)

        merge_keys = ["date", "track", "race_number", "greyhound"]
        available = [k for k in merge_keys if k in df.columns and k in form_df.columns]
        if available:
            df = df.merge(form_df[available + self._form_columns()],
                          on=available, how="left", suffixes=("", "_form"))
            drop = [c for c in df.columns if c.endswith("_form")]
            df = df.drop(columns=drop, errors="ignore")

        return df

    def _compute_form_for_dog(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute form features for a single greyhound's history."""
        group = group.copy()
        n = len(group)

        cols = self._form_columns()
        for col in cols:
            group[col] = np.nan

        dates = group["date"].values

        for i in range(1, n):
            # Get prior runs (up to max_lookback)
            start = max(0, i - self.max_lookback)
            prior = group.iloc[start:i]

            if prior.empty:
                continue

            # Calculate time-decay weights
            current_date = pd.Timestamp(dates[i])
            prior_dates = pd.to_datetime(prior["date"])
            days_ago = (current_date - prior_dates).dt.total_seconds() / 86400
            weights = np.exp(-np.log(2) * days_ago.values / self.half_life_days)
            weights = weights / weights.sum()  # normalise

            # Weighted average position
            if "finish_position" in prior.columns:
                pos = prior["finish_position"].values
                valid = ~np.isnan(pos)
                if valid.any():
                    group.iloc[i, group.columns.get_loc("form_weighted_pos")] = (
                        np.average(pos[valid], weights=weights[valid])
                    )

            # Weighted average time (for same distance if possible)
            if "finish_time" in prior.columns:
                times = prior["finish_time"].values
                valid = ~np.isnan(times)
                if valid.any():
                    group.iloc[i, group.columns.get_loc("form_weighted_time")] = (
                        np.average(times[valid], weights=weights[valid])
                    )

            # Win rate (weighted)
            if "won" in prior.columns:
                wins = prior["won"].values
                valid = ~np.isnan(wins)
                if valid.any():
                    group.iloc[i, group.columns.get_loc("form_weighted_win_rate")] = (
                        np.average(wins[valid], weights=weights[valid])
                    )

            # Form trajectory: is the dog improving or declining?
            if "finish_position" in prior.columns and len(prior) >= 3:
                recent_pos = prior["finish_position"].tail(3).values
                valid = ~np.isnan(recent_pos)
                if valid.sum() >= 2:
                    # Negative slope = improving (positions getting lower/better)
                    x = np.arange(valid.sum())
                    slope = np.polyfit(x, recent_pos[valid], 1)[0]
                    group.iloc[i, group.columns.get_loc("form_trajectory")] = -slope

            # Consistency (std of recent positions, lower = more consistent)
            if "finish_position" in prior.columns:
                pos = prior["finish_position"].dropna()
                if len(pos) >= 2:
                    group.iloc[i, group.columns.get_loc("form_consistency")] = pos.std()

            # Days since last run
            last_date = pd.Timestamp(dates[i - 1])
            group.iloc[i, group.columns.get_loc("form_days_off")] = (
                (current_date - last_date).total_seconds() / 86400
            )

            # Number of runs in form window
            group.iloc[i, group.columns.get_loc("form_run_count")] = len(prior)

        return group

    @staticmethod
    def _form_columns() -> list[str]:
        return [
            "form_weighted_pos",
            "form_weighted_time",
            "form_weighted_win_rate",
            "form_trajectory",
            "form_consistency",
            "form_days_off",
            "form_run_count",
        ]
