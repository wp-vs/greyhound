"""Track and trap bias modelling.

Different tracks have significant biases based on trap/box position.
For example, inside traps (1-2) often have an advantage on tight bends,
while outside traps (5-6) may benefit on wider tracks.

This module estimates trap bias per track from historical data.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrapBiasModel:
    """Model trap (starting box) bias at each track.

    Estimates the probability adjustment for each trap at each track
    based on historical win/place rates vs expected rates.
    """

    def __init__(self, min_samples: int = 50):
        """
        Args:
            min_samples: Minimum sample size per trap/track before
                         applying bias adjustment (to avoid noise).
        """
        self.min_samples = min_samples
        self._bias_table: pd.DataFrame | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "TrapBiasModel":
        """Fit trap bias model from historical results.

        Args:
            df: Results DataFrame with track, trap, won, placed columns.
        """
        required = ["track", "trap", "won"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        data = df.dropna(subset=required)

        # Count runners per race to get expected win rate
        if "race_id" in data.columns:
            runners_per_race = data.groupby("race_id")["trap"].count()
            data = data.merge(
                runners_per_race.rename("n_runners"),
                left_on="race_id",
                right_index=True,
                how="left",
            )
            data["expected_win_rate"] = 1.0 / data["n_runners"]
        else:
            data["expected_win_rate"] = 1.0 / 6.0  # assume 6 runners

        # Calculate actual vs expected win rates by track+trap
        bias = (
            data.groupby(["track", "trap"])
            .agg(
                wins=("won", "sum"),
                runs=("won", "count"),
                expected_wr=("expected_win_rate", "mean"),
            )
            .reset_index()
        )

        bias["actual_wr"] = bias["wins"] / bias["runs"]
        bias["bias"] = bias["actual_wr"] - bias["expected_wr"]

        # Only keep where we have sufficient data
        bias["significant"] = bias["runs"] >= self.min_samples
        bias.loc[~bias["significant"], "bias"] = 0.0

        # Also compute place bias if available
        if "placed" in data.columns:
            place_bias = (
                data.groupby(["track", "trap"])
                .agg(places=("placed", "sum"), runs_p=("placed", "count"))
                .reset_index()
            )
            place_bias["actual_pr"] = place_bias["places"] / place_bias["runs_p"]
            place_bias["expected_pr"] = 0.5  # ~3/6
            place_bias["place_bias"] = place_bias["actual_pr"] - place_bias["expected_pr"]
            place_bias.loc[place_bias["runs_p"] < self.min_samples, "place_bias"] = 0.0

            bias = bias.merge(
                place_bias[["track", "trap", "place_bias"]],
                on=["track", "trap"],
                how="left",
            )

        self._bias_table = bias
        self._fitted = True

        n_significant = bias["significant"].sum()
        logger.info(
            f"Trap bias model fitted: {len(bias)} track/trap combos, "
            f"{n_significant} with significant bias"
        )
        return self

    def get_bias(self, track: str, trap: int) -> float:
        """Get the win-rate bias for a specific track and trap.

        Returns:
            Bias value (positive = trap has advantage, negative = disadvantage).
        """
        if not self._fitted or self._bias_table is None:
            return 0.0

        match = self._bias_table[
            (self._bias_table["track"] == track) & (self._bias_table["trap"] == trap)
        ]

        if match.empty:
            return 0.0

        return float(match["bias"].iloc[0])

    def add_bias_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trap bias features to a DataFrame.

        Args:
            df: DataFrame with track and trap columns.

        Returns:
            DataFrame with trap_bias and trap_place_bias columns added.
        """
        if not self._fitted or self._bias_table is None:
            df["trap_bias"] = 0.0
            return df

        df = df.copy()
        df = df.merge(
            self._bias_table[["track", "trap", "bias", "place_bias"]].rename(
                columns={"bias": "trap_bias", "place_bias": "trap_place_bias"}
            ),
            on=["track", "trap"],
            how="left",
        )
        df["trap_bias"] = df["trap_bias"].fillna(0.0)
        if "trap_place_bias" in df.columns:
            df["trap_place_bias"] = df["trap_place_bias"].fillna(0.0)

        return df

    @property
    def bias_table(self) -> pd.DataFrame | None:
        """Return the full bias table for inspection."""
        return self._bias_table
