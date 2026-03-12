"""Market-implied probability extraction and value detection.

Bookmaker odds and exchange prices encode substantial information
about race outcomes. This module:

1. Converts odds to implied probabilities
2. Strips out the overround (bookmaker margin)
3. Compares model probabilities to market probabilities
4. Identifies "value" bets where the model disagrees with the market
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fractional_to_decimal(fractional: str) -> float:
    """Convert fractional odds to decimal.

    Examples:
        "5/2" -> 3.50
        "3/1" -> 4.00
        "evens" -> 2.00
        "1/2" -> 1.50
    """
    fractional = fractional.strip().lower()
    if fractional in ("evs", "evens"):
        return 2.0

    parts = fractional.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid fractional odds: {fractional}")

    num, den = int(parts[0]), int(parts[1])
    return round(num / den + 1, 4)


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def strip_overround(
    implied_probs: np.ndarray,
    method: str = "multiplicative",
) -> np.ndarray:
    """Remove bookmaker overround to get true probabilities.

    The overround is the amount by which implied probabilities
    sum to more than 1 (the bookmaker's margin).

    Args:
        implied_probs: Raw implied probabilities (sum > 1).
        method: How to remove overround:
            - "multiplicative": Scale all probs proportionally (simplest)
            - "power": Shin method — adjust using a power function
                       (better for favourite-longshot bias)
            - "additive": Subtract equal amount from each prob

    Returns:
        Adjusted probabilities summing to 1.
    """
    total = implied_probs.sum()
    if total <= 0:
        return np.full_like(implied_probs, 1.0 / len(implied_probs))

    overround = total - 1.0

    if method == "multiplicative":
        return implied_probs / total

    elif method == "additive":
        adj = overround / len(implied_probs)
        adjusted = implied_probs - adj
        adjusted = np.maximum(adjusted, 0.001)
        return adjusted / adjusted.sum()

    elif method == "power":
        # Shin's method: solve for z where sum of p_i^z = 1
        # Use binary search
        lo, hi = 0.5, 2.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if np.sum(implied_probs ** mid) > 1.0:
                lo = mid
            else:
                hi = mid

        adjusted = implied_probs ** ((lo + hi) / 2)
        return adjusted / adjusted.sum()

    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_overround(implied_probs: np.ndarray) -> float:
    """Calculate the bookmaker overround (margin).

    Returns:
        Overround as a percentage (e.g., 0.15 = 15% margin).
    """
    return implied_probs.sum() - 1.0


class ValueDetector:
    """Identify value bets by comparing model vs market probabilities.

    A "value" bet exists when the model's estimated probability
    exceeds the market-implied probability by a sufficient margin.
    """

    def __init__(self, min_edge: float = 0.05, min_prob: float = 0.10):
        """
        Args:
            min_edge: Minimum probability edge required (model - market).
            min_prob: Minimum model probability to consider (filters longshots).
        """
        self.min_edge = min_edge
        self.min_prob = min_prob

    def find_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find value bets in a set of races.

        Args:
            df: DataFrame with columns:
                - predicted_win_prob: model's probability
                - sp_decimal: starting price decimal odds
                - (optional) sp_implied_prob: market implied probability

        Returns:
            DataFrame of value selections with edge and expected_value columns.
        """
        df = df.copy()

        # Calculate market implied prob if not present
        if "sp_implied_prob" not in df.columns and "sp_decimal" in df.columns:
            df["sp_implied_prob"] = 1.0 / df["sp_decimal"]

        if "predicted_win_prob" not in df.columns or "sp_implied_prob" not in df.columns:
            logger.warning("Missing probability columns")
            return pd.DataFrame()

        # Strip overround within each race
        if "race_id" in df.columns:
            df["market_prob"] = df.groupby("race_id")["sp_implied_prob"].transform(
                lambda x: strip_overround(x.values, method="multiplicative")
            )
        else:
            df["market_prob"] = df["sp_implied_prob"]

        # Calculate edge
        df["edge"] = df["predicted_win_prob"] - df["market_prob"]

        # Expected value: (model_prob * (odds - 1)) - (1 - model_prob)
        if "sp_decimal" in df.columns:
            df["expected_value"] = (
                df["predicted_win_prob"] * (df["sp_decimal"] - 1)
                - (1 - df["predicted_win_prob"])
            )

        # Filter to value selections
        value = df[
            (df["edge"] >= self.min_edge) &
            (df["predicted_win_prob"] >= self.min_prob)
        ].copy()

        value = value.sort_values("edge", ascending=False)

        logger.info(
            f"Found {len(value)} value selections from {len(df)} runners "
            f"(edge >= {self.min_edge:.1%})"
        )

        return value

    @staticmethod
    def kelly_fraction(
        prob: float,
        decimal_odds: float,
        fraction: float = 0.25,
    ) -> float:
        """Calculate Kelly criterion stake as fraction of bankroll.

        Args:
            prob: Model's estimated win probability.
            decimal_odds: Decimal odds available.
            fraction: Fraction of full Kelly to use (0.25 = quarter Kelly,
                      recommended for noisy estimates).

        Returns:
            Recommended stake as fraction of bankroll (0 if no value).
        """
        if prob <= 0 or decimal_odds <= 1:
            return 0.0

        # Kelly: f = (bp - q) / b
        # where b = decimal_odds - 1, p = prob, q = 1 - prob
        b = decimal_odds - 1
        q = 1 - prob
        kelly = (b * prob - q) / b

        if kelly <= 0:
            return 0.0

        return kelly * fraction
