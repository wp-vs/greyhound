"""Data cleaning and feature engineering pipeline."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .schema import GRADE_NUMERIC

logger = logging.getLogger(__name__)


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw results data.

    - Remove incomplete records
    - Standardise formats
    - Add derived columns
    """
    df = df.copy()

    # Ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Remove rows without essential fields
    essential = ["date", "track", "greyhound", "finish_position"]
    for col in essential:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # Standardise track names
    if "track" in df.columns:
        df["track"] = df["track"].str.strip().str.title()

    # Standardise greyhound names
    if "greyhound" in df.columns:
        df["greyhound"] = df["greyhound"].str.strip().str.title()

    # Ensure numeric types
    numeric_cols = ["trap", "finish_position", "distance", "finish_time",
                    "sectional_time", "sp_decimal", "weight", "btn"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add numeric grade
    if "grade" in df.columns:
        df["grade_numeric"] = df["grade"].map(GRADE_NUMERIC)

    # Add win flag
    if "finish_position" in df.columns:
        df["won"] = (df["finish_position"] == 1).astype(int)
        df["placed"] = (df["finish_position"] <= 3).astype(int)

    # Add implied probability from SP
    if "sp_decimal" in df.columns:
        df["sp_implied_prob"] = 1.0 / df["sp_decimal"]

    df = df.sort_values(["date", "track", "race_number", "finish_position"])
    df = df.reset_index(drop=True)

    logger.info(f"Cleaned data: {len(df)} rows, {df['greyhound'].nunique()} greyhounds")
    return df


def build_race_features(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Build per-runner features from historical results.

    For each runner in each race, computes features from their
    prior runs (up to `lookback` previous races).

    Args:
        df: Cleaned results DataFrame, sorted by date.
        lookback: Number of previous runs to consider.

    Returns:
        DataFrame with feature columns added.
    """
    df = df.copy()
    df = df.sort_values(["date", "track", "race_number"]).reset_index(drop=True)

    # Group by greyhound and compute rolling stats
    features = []

    for name, group in df.groupby("greyhound"):
        group = group.sort_values("date").reset_index(drop=True)
        feat = _compute_runner_features(group, lookback)
        features.append(feat)

    if not features:
        return df

    feat_df = pd.concat(features, ignore_index=True)

    # Merge back on original index
    merge_cols = ["date", "track", "race_number", "greyhound"]
    available = [c for c in merge_cols if c in df.columns and c in feat_df.columns]
    df = df.merge(feat_df, on=available, how="left", suffixes=("", "_feat"))

    # Drop duplicate columns from merge
    drop_cols = [c for c in df.columns if c.endswith("_feat")]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def _compute_runner_features(group: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Compute rolling features for a single greyhound."""
    group = group.copy()

    # Win/place rates (expanding, shifted to avoid leakage)
    if "won" in group.columns:
        group["career_wins"] = group["won"].shift(1).expanding().sum()
        group["career_runs"] = group["won"].shift(1).expanding().count()
        group["career_win_rate"] = group["career_wins"] / group["career_runs"].clip(lower=1)

    if "placed" in group.columns:
        group["career_place_rate"] = (
            group["placed"].shift(1).expanding().mean()
        )

    # Recent form: rolling stats over last N runs
    if "finish_position" in group.columns:
        shifted_pos = group["finish_position"].shift(1)
        group["avg_position_last5"] = shifted_pos.rolling(min(5, lookback), min_periods=1).mean()
        group["avg_position_last3"] = shifted_pos.rolling(min(3, lookback), min_periods=1).mean()
        group["best_position_last5"] = shifted_pos.rolling(min(5, lookback), min_periods=1).min()

    # Time-based features
    if "finish_time" in group.columns:
        shifted_time = group["finish_time"].shift(1)
        group["avg_time_last5"] = shifted_time.rolling(min(5, lookback), min_periods=1).mean()
        group["best_time_last5"] = shifted_time.rolling(min(5, lookback), min_periods=1).min()
        group["time_std_last5"] = shifted_time.rolling(min(5, lookback), min_periods=2).std()

    # Days since last run
    if "date" in group.columns:
        group["days_since_last"] = group["date"].diff().dt.days

    # Distance experience
    if "distance" in group.columns:
        group["runs_at_distance"] = group.groupby("distance").cumcount()

    # Track experience
    if "track" in group.columns:
        group["runs_at_track"] = group.groupby("track").cumcount()

    # Trap stats
    if "trap" in group.columns and "won" in group.columns:
        group["runs_from_trap"] = group.groupby("trap").cumcount()

    # Grade momentum (are they going up or down in class?)
    if "grade_numeric" in group.columns:
        shifted_grade = group["grade_numeric"].shift(1)
        group["grade_change"] = group["grade_numeric"] - shifted_grade
        group["avg_grade_last3"] = shifted_grade.rolling(3, min_periods=1).mean()

    return group


def create_race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add race-level context features (relative to other runners)."""
    df = df.copy()

    # Within-race rankings
    for col in ["sp_implied_prob", "avg_time_last5", "career_win_rate"]:
        if col in df.columns:
            rank_col = f"{col}_rank"
            df[rank_col] = df.groupby("race_id")[col].rank(method="min", ascending=False)

    # Market position
    if "sp_decimal" in df.columns:
        df["is_favourite"] = (
            df.groupby("race_id")["sp_decimal"]
            .transform(lambda x: x == x.min())
            .astype(int)
        )

    return df
