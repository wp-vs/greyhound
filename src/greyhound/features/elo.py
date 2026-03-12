"""Elo rating system adapted for greyhound racing.

Each greyhound has a dynamic rating updated after every race.
Ratings are updated based on finishing order relative to
the other runners, using a multi-player Elo extension.

Simple to implement, surprisingly predictive, and excellent
as a single feature or as part of an ensemble.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EloRating:
    """Multi-player Elo rating system for greyhound racing.

    Each runner is compared pairwise against every other runner in the
    same race. If a runner finishes ahead, that counts as a "win" in
    the pairwise comparison. Ratings are updated accordingly.
    """

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 20.0,
        distance_weight: bool = True,
    ):
        """
        Args:
            initial_rating: Starting Elo for new greyhounds.
            k_factor: Controls how quickly ratings change.
                      Higher = more responsive but noisier.
            distance_weight: If True, weight updates by finishing margin.
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.distance_weight = distance_weight
        self.ratings: dict[str, float] = {}
        self._history: list[dict] = []

    def get_rating(self, name: str) -> float:
        """Get current rating for a greyhound."""
        return self.ratings.get(name, self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score (win probability) of A vs B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_race(self, results: list[dict]) -> dict[str, float]:
        """Update ratings based on a single race result.

        Args:
            results: List of dicts with keys:
                - name: greyhound name
                - position: finishing position (1=winner)
                - btn: beaten distance (optional, for margin weighting)

        Returns:
            Dict of {name: new_rating} for all runners.
        """
        n = len(results)
        if n < 2:
            return {}

        # Get current ratings
        names = [r["name"] for r in results]
        current = {r["name"]: self.get_rating(r["name"]) for r in results}

        updates = {name: 0.0 for name in names}

        # Pairwise comparisons
        for i in range(n):
            for j in range(i + 1, n):
                ri = results[i]
                rj = results[j]

                # Actual score: 1 if i beat j, 0 otherwise
                if ri["position"] < rj["position"]:
                    actual_i = 1.0
                elif ri["position"] > rj["position"]:
                    actual_i = 0.0
                else:
                    actual_i = 0.5  # dead heat

                expected_i = self.expected_score(current[ri["name"]], current[rj["name"]])

                # Scale K by number of comparisons
                k = self.k_factor / (n - 1)

                # Optional margin weighting
                weight = 1.0
                if self.distance_weight:
                    btn_i = ri.get("btn", 0) or 0
                    btn_j = rj.get("btn", 0) or 0
                    margin = abs(btn_j - btn_i)
                    weight = 1.0 + min(margin, 10.0) / 10.0

                delta = k * weight * (actual_i - expected_i)
                updates[ri["name"]] += delta
                updates[rj["name"]] -= delta

        # Apply updates
        new_ratings = {}
        for name in names:
            new_rating = current[name] + updates[name]
            self.ratings[name] = new_rating
            new_ratings[name] = new_rating

        return new_ratings

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a full results DataFrame and add Elo ratings.

        Iterates through races chronologically, updating ratings
        and recording each runner's rating BEFORE the race (for prediction).

        Args:
            df: Results DataFrame sorted by date, with columns:
                race_id, greyhound, finish_position, btn (optional).

        Returns:
            DataFrame with elo_rating column added (pre-race rating).
        """
        df = df.copy()
        df = df.sort_values(["date", "track", "race_number"]).reset_index(drop=True)

        elo_ratings = pd.Series(np.nan, index=df.index, dtype=float)

        # Process each race
        if "race_id" not in df.columns:
            df["race_id"] = (
                df["date"].astype(str) + "_" + df["track"] + "_" + df["race_number"].astype(str)
            )

        for race_id, race_group in df.groupby("race_id", sort=False):
            # Record pre-race ratings
            for idx, row in race_group.iterrows():
                elo_ratings.loc[idx] = self.get_rating(row["greyhound"])

            # Update ratings
            results = []
            for _, row in race_group.iterrows():
                results.append({
                    "name": row["greyhound"],
                    "position": row["finish_position"],
                    "btn": row.get("btn"),
                })

            self.update_race(results)

        df["elo_rating"] = elo_ratings
        return df

    def predict_race(self, runners: list[str]) -> dict[str, float]:
        """Predict win probabilities for a set of runners.

        Uses the Bradley-Terry model extension: each runner's
        win probability is proportional to 10^(rating/400).

        Args:
            runners: List of greyhound names.

        Returns:
            Dict of {name: win_probability}.
        """
        strengths = {}
        for name in runners:
            rating = self.get_rating(name)
            strengths[name] = 10.0 ** (rating / 400.0)

        total = sum(strengths.values())
        return {name: s / total for name, s in strengths.items()}

    def top_rated(self, n: int = 20) -> list[tuple[str, float]]:
        """Get the top-rated greyhounds."""
        sorted_ratings = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_ratings[:n]
