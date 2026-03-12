"""End-to-end pipeline for greyhound race prediction.

Orchestrates:
1. Data loading and cleaning
2. Feature engineering (speed ratings, form, Elo, trap bias, class adjustment)
3. Model training (multiple model types)
4. Calibration
5. Ensemble combination
6. Evaluation
7. Value detection
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .data.preprocessing import build_race_features, clean_results, create_race_context_features
from .data.scraper import GreyhoundScraper, HistoricalDataLoader
from .evaluation.metrics import evaluate_model, print_evaluation_report
from .features.class_adjustment import ClassAdjuster
from .features.elo import EloRating
from .features.form import FormCalculator
from .features.speed_ratings import SectionalAnalyser, SpeedRatingCalculator
from .features.track_bias import TrapBiasModel
from .market.implied_probability import ValueDetector
from .models.base import FEATURE_COLUMNS
from .models.calibration import ProbabilityCalibrator, calibrate_within_race
from .models.ensemble import EnsembleModel
from .models.gradient_boosting import GradientBoostingModel, RankingModel
from .models.logistic import LogisticModel
from .models.ordered import OrderedProbitModel
from .models.survival import SurvivalModel

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end greyhound race prediction pipeline."""

    def __init__(
        self,
        data_dir: str = "data",
        model_dir: str = "models/saved",
        val_split_date: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Directory containing raw data files.
            model_dir: Directory to save/load trained models.
            val_split_date: Date string (YYYY-MM-DD) to split train/validation.
                           If None, uses last 20% of data.
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.val_split_date = val_split_date

        # Components
        self.speed_calculator = SpeedRatingCalculator()
        self.form_calculator = FormCalculator()
        self.trap_bias_model = TrapBiasModel()
        self.class_adjuster = ClassAdjuster()
        self.elo = EloRating()
        self.calibrator = ProbabilityCalibrator(method="isotonic")
        self.value_detector = ValueDetector()

        # Models
        self.models: dict[str, object] = {}
        self.ensemble: EnsembleModel | None = None

    def load_data(self) -> pd.DataFrame:
        """Load all available data."""
        loader = HistoricalDataLoader(str(self.data_dir))
        df = loader.load_all()
        if df.empty:
            logger.warning("No data loaded. Run scraper first.")
        return df

    def scrape_data(
        self,
        start_date: date,
        end_date: date,
        tracks: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Scrape new data and save to disk."""
        scraper = GreyhoundScraper(output_dir=str(self.data_dir / "raw"))
        races = scraper.scrape_results(start_date, end_date, tracks)
        if races:
            filename = f"results_{start_date}_{end_date}.parquet"
            scraper.save_results(races, filename)
        return scraper.to_dataframe(races)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline."""
        logger.info("Starting feature engineering...")

        # 1. Clean data
        df = clean_results(df)

        # 2. Build race ID if not present
        if "race_id" not in df.columns:
            df["race_id"] = (
                df["date"].dt.strftime("%Y-%m-%d") + "_"
                + df["track"] + "_"
                + df["race_number"].astype(str)
            )

        # 3. Speed ratings
        logger.info("  Computing speed ratings...")
        self.speed_calculator.fit(df)
        df["speed_rating"] = self.speed_calculator.calculate(df)

        # 4. Sectional analysis
        logger.info("  Analysing sectionals...")
        analyser = SectionalAnalyser()
        df = analyser.calculate_pace_profile(df)

        # 5. Class adjustment
        logger.info("  Fitting class adjustments...")
        self.class_adjuster.fit(df)
        df = self.class_adjuster.add_adjusted_times(df)

        # 6. Form calculation
        logger.info("  Computing form features...")
        df = self.form_calculator.calculate_form(df)

        # 7. Basic rolling features
        logger.info("  Computing rolling features...")
        df = build_race_features(df)

        # 8. Trap bias
        logger.info("  Fitting trap bias model...")
        self.trap_bias_model.fit(df)
        df = self.trap_bias_model.add_bias_features(df)

        # 9. Elo ratings
        logger.info("  Computing Elo ratings...")
        df = self.elo.process_dataframe(df)

        # 10. Race context features
        logger.info("  Adding race context features...")
        df = create_race_context_features(df)

        logger.info(f"Feature engineering complete: {df.shape}")
        return df

    def split_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split into training and validation sets.

        Uses temporal split (train on earlier data, validate on later)
        to avoid lookahead bias.
        """
        df = df.sort_values("date").reset_index(drop=True)

        if self.val_split_date:
            split = pd.Timestamp(self.val_split_date)
        else:
            # Use last 20% by date
            dates = df["date"].sort_values().unique()
            split_idx = int(len(dates) * 0.8)
            split = dates[split_idx]

        train = df[df["date"] < split].copy()
        val = df[df["date"] >= split].copy()

        logger.info(
            f"Train: {len(train)} rows (up to {split}), "
            f"Val: {len(val)} rows (from {split})"
        )
        return train, val

    def train_models(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> dict[str, object]:
        """Train all model types."""
        logger.info("Training models...")

        # 1. Logistic regression
        logger.info("  Training logistic regression...")
        logistic = LogisticModel()
        try:
            logistic.fit(train_df)
            self.models["logistic"] = logistic
        except Exception as e:
            logger.error(f"  Logistic model failed: {e}")

        # 2. Ordered probit
        logger.info("  Training ordered probit...")
        ordered = OrderedProbitModel()
        try:
            ordered.fit(train_df)
            self.models["ordered_probit"] = ordered
        except Exception as e:
            logger.error(f"  Ordered probit failed: {e}")

        # 3. XGBoost
        logger.info("  Training XGBoost...")
        xgb_model = GradientBoostingModel(engine="xgboost")
        try:
            xgb_model.fit(train_df, val_df)
            self.models["xgboost"] = xgb_model
        except Exception as e:
            logger.error(f"  XGBoost failed: {e}")

        # 4. LightGBM
        logger.info("  Training LightGBM...")
        lgb_model = GradientBoostingModel(engine="lightgbm")
        try:
            lgb_model.fit(train_df, val_df)
            self.models["lightgbm"] = lgb_model
        except Exception as e:
            logger.error(f"  LightGBM failed: {e}")

        # 5. Survival model
        logger.info("  Training survival model...")
        survival = SurvivalModel()
        try:
            survival.fit(train_df)
            self.models["survival"] = survival
        except Exception as e:
            logger.error(f"  Survival model failed: {e}")

        # 6. Ranking model
        logger.info("  Training ranking model...")
        ranking = RankingModel()
        try:
            ranking.fit(train_df)
            self.models["ranking"] = ranking
        except Exception as e:
            logger.error(f"  Ranking model failed: {e}")

        logger.info(f"Trained {len(self.models)} models successfully")
        return self.models

    def build_ensemble(
        self, val_df: pd.DataFrame
    ) -> EnsembleModel:
        """Build an optimised ensemble from trained models."""
        fitted_models = [m for m in self.models.values() if hasattr(m, "predict_proba")]

        if len(fitted_models) < 2:
            logger.warning("Need at least 2 models for ensemble")
            if fitted_models:
                self.ensemble = EnsembleModel(fitted_models)
                self.ensemble._fitted = True
                return self.ensemble
            raise RuntimeError("No models available for ensemble")

        self.ensemble = EnsembleModel(fitted_models, method="weighted_average")
        self.ensemble.optimize_weights(val_df, metric="log_loss")
        self.ensemble._fitted = True

        return self.ensemble

    def calibrate(
        self, val_df: pd.DataFrame, model_name: str = "ensemble"
    ) -> ProbabilityCalibrator:
        """Fit probability calibration on validation data."""
        if model_name == "ensemble" and self.ensemble:
            probs = self.ensemble.predict_proba(val_df)
        elif model_name in self.models:
            probs = self.models[model_name].predict_proba(val_df)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        y_true = val_df["won"].values
        self.calibrator.fit(y_true, probs)
        return self.calibrator

    def predict(self, df: pd.DataFrame, calibrate: bool = True) -> pd.DataFrame:
        """Generate predictions for new data.

        Args:
            df: DataFrame with feature columns.
            calibrate: Whether to apply probability calibration.

        Returns:
            DataFrame with predicted_win_prob column added.
        """
        if self.ensemble and self.ensemble._fitted:
            probs = self.ensemble.predict_proba(df)
        elif self.models:
            # Use best single model
            model = list(self.models.values())[0]
            probs = model.predict_proba(df)
        else:
            raise RuntimeError("No trained models available")

        if calibrate and self.calibrator._fitted:
            probs = self.calibrator.calibrate(probs)

        # Normalise within each race
        df = df.copy()
        df["predicted_win_prob"] = probs

        if "race_id" in df.columns:
            df["predicted_win_prob"] = df.groupby("race_id")[
                "predicted_win_prob"
            ].transform(lambda x: calibrate_within_race(x.values))

        df["predicted_rank"] = df.groupby("race_id")["predicted_win_prob"].rank(
            ascending=False, method="min"
        )

        return df

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Evaluate predictions against actual outcomes."""
        metrics = evaluate_model(df)
        report = print_evaluation_report(metrics)
        logger.info("\n" + report)
        return metrics

    def find_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find value bets in predicted data."""
        return self.value_detector.find_value(df)

    def run(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Run the full pipeline end-to-end.

        Args:
            df: Optional pre-loaded DataFrame. If None, loads from disk.

        Returns:
            Dict with results including metrics and value bets.
        """
        # 1. Load data
        if df is None:
            df = self.load_data()
            if df.empty:
                return {"error": "No data available"}

        # 2. Feature engineering
        df = self.engineer_features(df)

        # 3. Split
        train_df, val_df = self.split_data(df)

        # 4. Train models
        self.train_models(train_df, val_df)

        # 5. Build ensemble
        self.build_ensemble(val_df)

        # 6. Calibrate
        self.calibrate(val_df)

        # 7. Predict on validation set
        val_df = self.predict(val_df)

        # 8. Evaluate
        metrics = self.evaluate(val_df)

        # 9. Find value bets
        value_bets = self.find_value(val_df)

        return {
            "metrics": metrics,
            "value_bets": value_bets,
            "validation_data": val_df,
            "n_models": len(self.models),
        }

    def save(self, path: Optional[str] = None):
        """Save the trained pipeline to disk."""
        save_path = Path(path) if path else self.model_dir / "pipeline.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "speed_calculator": self.speed_calculator,
            "form_calculator": self.form_calculator,
            "trap_bias_model": self.trap_bias_model,
            "class_adjuster": self.class_adjuster,
            "elo": self.elo,
            "calibrator": self.calibrator,
            "models": self.models,
            "ensemble": self.ensemble,
        }
        joblib.dump(state, save_path)
        logger.info(f"Pipeline saved to {save_path}")

    def load(self, path: Optional[str] = None):
        """Load a trained pipeline from disk."""
        load_path = Path(path) if path else self.model_dir / "pipeline.joblib"
        state = joblib.load(load_path)

        self.speed_calculator = state["speed_calculator"]
        self.form_calculator = state["form_calculator"]
        self.trap_bias_model = state["trap_bias_model"]
        self.class_adjuster = state["class_adjuster"]
        self.elo = state["elo"]
        self.calibrator = state["calibrator"]
        self.models = state["models"]
        self.ensemble = state["ensemble"]

        logger.info(f"Pipeline loaded from {load_path}")
