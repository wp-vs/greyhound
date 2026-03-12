"""Model evaluation metrics for greyhound race prediction.

Includes both standard ML metrics and racing-specific measures:
- Log loss, Brier score (probability accuracy)
- ROI / yield (profitability if betting model selections)
- Calibration metrics
- Rank correlation (does model rank runners correctly?)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


def evaluate_model(
    df: pd.DataFrame,
    prob_col: str = "predicted_win_prob",
    target_col: str = "won",
) -> dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Args:
        df: DataFrame with predictions and outcomes.
        prob_col: Column name for predicted probabilities.
        target_col: Column name for binary outcome.

    Returns:
        Dict of metric_name: value.
    """
    y_true = df[target_col].values
    y_prob = df[prob_col].values

    # Clip probabilities
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)

    metrics = {}

    # Standard classification metrics
    metrics["log_loss"] = log_loss(y_true, y_prob)
    metrics["brier_score"] = brier_score_loss(y_true, y_prob)

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    # Racing-specific metrics
    # Win rate of model's top pick per race
    if "race_id" in df.columns:
        top_picks = df.loc[df.groupby("race_id")[prob_col].idxmax()]
        metrics["top_pick_win_rate"] = top_picks[target_col].mean()
        metrics["top_pick_place_rate"] = (top_picks["finish_position"] <= 3).mean()

        # Rank correlation per race
        from scipy.stats import spearmanr
        correlations = []
        for _, race in df.groupby("race_id"):
            if len(race) >= 3:
                corr, _ = spearmanr(
                    race[prob_col].values,
                    -race["finish_position"].values,  # negate so higher prob = lower position
                )
                if not np.isnan(corr):
                    correlations.append(corr)

        if correlations:
            metrics["mean_rank_correlation"] = np.mean(correlations)

    # Profitability metrics (if odds available)
    if "sp_decimal" in df.columns:
        # Flat staking on top pick per race
        if "race_id" in df.columns:
            top_picks = df.loc[df.groupby("race_id")[prob_col].idxmax()]
            returns = np.where(
                top_picks[target_col] == 1,
                top_picks["sp_decimal"] - 1,
                -1,
            )
            metrics["top_pick_roi"] = returns.mean()
            metrics["top_pick_yield_pct"] = returns.mean() * 100

        # Value bet ROI (where model prob > implied prob)
        if "sp_implied_prob" in df.columns:
            value_mask = df[prob_col] > df["sp_implied_prob"]
            if value_mask.any():
                value_returns = np.where(
                    df.loc[value_mask, target_col] == 1,
                    df.loc[value_mask, "sp_decimal"] - 1,
                    -1,
                )
                metrics["value_bet_roi"] = value_returns.mean()
                metrics["value_bet_count"] = value_mask.sum()

    # Calibration metrics
    metrics["mean_predicted"] = y_prob.mean()
    metrics["mean_actual"] = y_true.mean()
    metrics["calibration_gap"] = abs(y_prob.mean() - y_true.mean())

    return metrics


def evaluate_by_group(
    df: pd.DataFrame,
    group_col: str,
    prob_col: str = "predicted_win_prob",
    target_col: str = "won",
) -> pd.DataFrame:
    """Evaluate model performance broken down by a grouping variable.

    Useful for checking if the model performs differently across
    tracks, grades, distances, etc.

    Args:
        df: DataFrame with predictions and outcomes.
        group_col: Column to group by (e.g., "track", "grade", "distance").
        prob_col: Predicted probability column.
        target_col: Target column.

    Returns:
        DataFrame with metrics per group.
    """
    results = []
    for group_val, group_df in df.groupby(group_col):
        if len(group_df) < 20:
            continue
        metrics = evaluate_model(group_df, prob_col, target_col)
        metrics[group_col] = group_val
        metrics["n_runners"] = len(group_df)
        results.append(metrics)

    return pd.DataFrame(results).set_index(group_col)


def print_evaluation_report(metrics: dict[str, float]) -> str:
    """Format evaluation metrics as a readable report."""
    lines = [
        "=" * 50,
        "MODEL EVALUATION REPORT",
        "=" * 50,
        "",
        "Probability Accuracy:",
        f"  Log Loss:           {metrics.get('log_loss', float('nan')):.4f}",
        f"  Brier Score:        {metrics.get('brier_score', float('nan')):.4f}",
        f"  ROC AUC:            {metrics.get('roc_auc', float('nan')):.4f}",
        "",
        "Prediction Quality:",
        f"  Top Pick Win Rate:  {metrics.get('top_pick_win_rate', float('nan')):.1%}",
        f"  Top Pick Place Rate:{metrics.get('top_pick_place_rate', float('nan')):.1%}",
        f"  Rank Correlation:   {metrics.get('mean_rank_correlation', float('nan')):.4f}",
        "",
        "Calibration:",
        f"  Mean Predicted:     {metrics.get('mean_predicted', float('nan')):.4f}",
        f"  Mean Actual:        {metrics.get('mean_actual', float('nan')):.4f}",
        f"  Calibration Gap:    {metrics.get('calibration_gap', float('nan')):.4f}",
        "",
    ]

    if "top_pick_roi" in metrics:
        lines.extend([
            "Profitability (flat staking):",
            f"  Top Pick ROI:       {metrics.get('top_pick_roi', float('nan')):.1%}",
            f"  Value Bet ROI:      {metrics.get('value_bet_roi', float('nan')):.1%}",
            f"  Value Bet Count:    {metrics.get('value_bet_count', 0):.0f}",
            "",
        ])

    lines.append("=" * 50)
    return "\n".join(lines)
