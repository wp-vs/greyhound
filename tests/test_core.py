"""Core tests for the greyhound prediction system."""

import numpy as np
import pandas as pd
import pytest

from greyhound.data.preprocessing import clean_results, build_race_features
from greyhound.data.schema import GRADE_NUMERIC
from greyhound.features.elo import EloRating
from greyhound.features.speed_ratings import SpeedRatingCalculator
from greyhound.features.track_bias import TrapBiasModel
from greyhound.features.form import FormCalculator
from greyhound.features.class_adjustment import ClassAdjuster
from greyhound.market.implied_probability import (
    fractional_to_decimal,
    decimal_to_implied_prob,
    strip_overround,
    ValueDetector,
)
from greyhound.models.calibration import ProbabilityCalibrator, calibrate_within_race


def make_sample_data(n_races=50, n_runners=6) -> pd.DataFrame:
    """Create synthetic race data for testing."""
    np.random.seed(42)
    rows = []
    for i in range(n_races):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i // 3)
        track = np.random.choice(["Romford", "Crayford", "Hove"])
        distance = np.random.choice([400, 480, 575])
        grade = np.random.choice(["A3", "A5", "A7"])

        # Generate random finishing order
        positions = np.random.permutation(n_runners) + 1
        base_time = 24.0 + distance / 100.0

        for j in range(n_runners):
            rows.append({
                "date": date,
                "track": track,
                "race_number": i % 12 + 1,
                "distance": distance,
                "grade": grade,
                "greyhound": f"Dog_{(i * n_runners + j) % 30}",
                "trap": j + 1,
                "finish_position": int(positions[j]),
                "finish_time": base_time + positions[j] * 0.3 + np.random.normal(0, 0.1),
                "sp_decimal": np.random.uniform(2.0, 15.0),
                "sectional_time": base_time * 0.35 + np.random.normal(0, 0.05),
            })

    return pd.DataFrame(rows)


class TestDataPreprocessing:
    def test_clean_results(self):
        df = make_sample_data()
        cleaned = clean_results(df)

        assert "won" in cleaned.columns
        assert "placed" in cleaned.columns
        assert "grade_numeric" in cleaned.columns
        assert "sp_implied_prob" in cleaned.columns
        assert cleaned["won"].sum() > 0

    def test_build_race_features(self):
        df = make_sample_data()
        df = clean_results(df)
        featured = build_race_features(df, lookback=5)

        assert "career_win_rate" in featured.columns
        assert "avg_position_last5" in featured.columns


class TestElo:
    def test_initial_rating(self):
        elo = EloRating()
        assert elo.get_rating("Unknown Dog") == 1500.0

    def test_update_race(self):
        elo = EloRating()
        results = [
            {"name": "Winner", "position": 1},
            {"name": "Second", "position": 2},
            {"name": "Third", "position": 3},
        ]
        new_ratings = elo.update_race(results)

        assert new_ratings["Winner"] > 1500
        assert new_ratings["Third"] < 1500

    def test_predict_race(self):
        elo = EloRating()
        elo.ratings = {"Fast Dog": 1600, "Slow Dog": 1400, "Average Dog": 1500}

        probs = elo.predict_race(["Fast Dog", "Slow Dog", "Average Dog"])

        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert probs["Fast Dog"] > probs["Average Dog"] > probs["Slow Dog"]

    def test_process_dataframe(self):
        df = make_sample_data(n_races=20)
        df = clean_results(df)
        elo = EloRating()
        result = elo.process_dataframe(df)

        assert "elo_rating" in result.columns
        assert result["elo_rating"].notna().any()


class TestSpeedRatings:
    def test_fit_and_calculate(self):
        df = make_sample_data()
        calc = SpeedRatingCalculator()
        calc.fit(df)
        ratings = calc.calculate(df)

        assert len(ratings) == len(df)
        assert ratings.notna().any()


class TestTrapBias:
    def test_fit_and_query(self):
        df = make_sample_data(n_races=200)
        df = clean_results(df)
        df["race_id"] = df["date"].astype(str) + "_" + df["track"] + "_" + df["race_number"].astype(str)

        model = TrapBiasModel(min_samples=10)
        model.fit(df)

        bias = model.get_bias("Romford", 1)
        assert isinstance(bias, float)

    def test_add_bias_features(self):
        df = make_sample_data(n_races=200)
        df = clean_results(df)
        df["race_id"] = df["date"].astype(str) + "_" + df["track"] + "_" + df["race_number"].astype(str)

        model = TrapBiasModel(min_samples=10)
        model.fit(df)
        result = model.add_bias_features(df)

        assert "trap_bias" in result.columns


class TestForm:
    def test_calculate_form(self):
        df = make_sample_data(n_races=30)
        df = clean_results(df)

        calc = FormCalculator(half_life_days=28)
        result = calc.calculate_form(df)

        assert "form_weighted_pos" in result.columns
        assert "form_trajectory" in result.columns


class TestClassAdjustment:
    def test_fit_and_adjust(self):
        df = make_sample_data(n_races=100)
        df = clean_results(df)

        adjuster = ClassAdjuster()
        adjuster.fit(df)
        result = adjuster.add_adjusted_times(df)

        assert "adjusted_time" in result.columns


class TestMarket:
    def test_fractional_to_decimal(self):
        assert fractional_to_decimal("5/2") == 3.5
        assert fractional_to_decimal("3/1") == 4.0
        assert fractional_to_decimal("1/2") == 1.5
        assert fractional_to_decimal("evens") == 2.0

    def test_decimal_to_implied_prob(self):
        assert decimal_to_implied_prob(2.0) == 0.5
        assert abs(decimal_to_implied_prob(4.0) - 0.25) < 1e-6

    def test_strip_overround(self):
        # 6 runner race, each at 4.5 => sum = 6/4.5 = 1.333
        probs = np.array([1/4.5] * 6)
        adjusted = strip_overround(probs, method="multiplicative")
        assert abs(adjusted.sum() - 1.0) < 1e-6

    def test_value_detector(self):
        df = pd.DataFrame({
            "race_id": ["R1"] * 3,
            "predicted_win_prob": [0.40, 0.35, 0.25],
            "sp_decimal": [4.0, 3.0, 5.0],
            "sp_implied_prob": [0.25, 0.333, 0.20],
            "won": [1, 0, 0],
            "finish_position": [1, 2, 3],
        })

        detector = ValueDetector(min_edge=0.05)
        value = detector.find_value(df)
        assert len(value) > 0


class TestCalibration:
    def test_platt_scaling(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.2, 1000)
        y_prob = np.random.beta(2, 8, 1000)

        cal = ProbabilityCalibrator(method="platt")
        cal.fit(y_true, y_prob)
        calibrated = cal.calibrate(y_prob)

        assert len(calibrated) == len(y_prob)
        assert all(0 <= p <= 1 for p in calibrated)

    def test_isotonic(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.2, 1000)
        y_prob = np.random.beta(2, 8, 1000)

        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(y_true, y_prob)
        calibrated = cal.calibrate(y_prob)

        assert len(calibrated) == len(y_prob)

    def test_calibrate_within_race(self):
        probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.05])
        normalised = calibrate_within_race(probs)
        assert abs(normalised.sum() - 1.0) < 1e-6
