"""
tests/test_pipeline.py
Unit tests for the Traffic Congestion Prediction pipeline.
Run with: pytest tests/test_pipeline.py -v
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_data, generate_preprocessing_summary
from src.feature_engineering import (
    add_temporal_features,
    add_cyclical_features,
    fit_historical_profiles,
    apply_historical_profiles,
    build_features,
)
from src.risk_scoring import (
    fit_risk_thresholds,
    calculate_congestion_metrics,
    add_risk_scores_to_df,
)
from src.evaluate import calculate_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_raw_df():
    """Minimal synthetic traffic dataframe that mimics the real dataset schema."""
    dates = pd.date_range("2023-01-01 00:00", periods=96, freq="1H")
    junctions = [1, 2, 3, 4] * 24
    vehicles = np.random.randint(5, 120, size=96)
    return pd.DataFrame({
        "DateTime": dates.tolist() * 1,
        "Junction": junctions,
        "Vehicles": vehicles,
        "ID": range(96),
    })


@pytest.fixture
def clean_df(sample_raw_df):
    return preprocess_data(sample_raw_df)


@pytest.fixture
def featured_df(clean_df):
    profiles = fit_historical_profiles(clean_df)
    return build_features(clean_df, profiles), profiles


# ── Test: Preprocessing ───────────────────────────────────────────────────────
class TestPreprocessing:
    def test_datetime_parsed(self, clean_df):
        assert pd.api.types.is_datetime64_any_dtype(clean_df["DateTime"])

    def test_no_negative_vehicles(self, clean_df):
        assert (clean_df["Vehicles"] >= 0).all()

    def test_shape_preserved(self, sample_raw_df, clean_df):
        assert len(clean_df) <= len(sample_raw_df)

    def test_summary_keys(self, sample_raw_df, clean_df):
        summary = generate_preprocessing_summary(sample_raw_df, clean_df)
        for key in ["raw_shape", "clean_shape", "min_date", "max_date", "junctions"]:
            assert key in summary


# ── Test: Feature Engineering ─────────────────────────────────────────────────
class TestFeatureEngineering:
    def test_temporal_columns(self, clean_df):
        df = add_temporal_features(clean_df)
        for col in ["Hour", "Day", "DayOfWeek", "Month", "Year", "IsWeekend", "IsRushHour", "TimeOfDay"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_cyclical_columns(self, clean_df):
        df = add_temporal_features(clean_df)
        df = add_cyclical_features(df)
        for col in ["Hour_sin", "Hour_cos", "DayOfWeek_sin", "DayOfWeek_cos", "Month_sin", "Month_cos"]:
            assert col in df.columns

    def test_cyclical_range(self, clean_df):
        df = add_temporal_features(clean_df)
        df = add_cyclical_features(df)
        assert df["Hour_sin"].between(-1.001, 1.001).all()
        assert df["Hour_cos"].between(-1.001, 1.001).all()

    def test_historical_profiles_keys(self, clean_df):
        profiles = fit_historical_profiles(clean_df)
        for key in ["overall_mean", "junction_mean", "j_hour_mean", "j_dow_hour_mean"]:
            assert key in profiles

    def test_historical_profile_application(self, clean_df):
        profiles = fit_historical_profiles(clean_df)
        df = build_features(clean_df, profiles)
        assert "Hist_Traffic_J_D_H" in df.columns
        assert "Hist_Traffic_J_H" in df.columns
        assert df["Hist_Traffic_J_D_H"].notna().all()

    def test_is_weekend_binary(self, clean_df):
        df = add_temporal_features(clean_df)
        assert set(df["IsWeekend"].unique()).issubset({0, 1})

    def test_rush_hour_binary(self, clean_df):
        df = add_temporal_features(clean_df)
        assert set(df["IsRushHour"].unique()).issubset({0, 1})


# ── Test: Risk Scoring ────────────────────────────────────────────────────────
class TestRiskScoring:
    def test_thresholds_per_junction(self, clean_df):
        thresholds = fit_risk_thresholds(clean_df)
        assert "global" in thresholds
        for j in clean_df["Junction"].unique():
            assert str(j) in thresholds

    def test_risk_score_range(self, clean_df):
        thresholds = fit_risk_thresholds(clean_df)
        for vehicles in [0, 10, 50, 200, 1000]:
            result = calculate_congestion_metrics(vehicles, 1, thresholds)
            assert 0.0 <= result["risk_score"] <= 100.0

    def test_congestion_levels(self, clean_df):
        thresholds = fit_risk_thresholds(clean_df)
        result = calculate_congestion_metrics(0, 1, thresholds)
        assert result["congestion_level"] in {"Low", "Medium", "High"}

    def test_low_volume_low_risk(self, clean_df):
        thresholds = fit_risk_thresholds(clean_df)
        result = calculate_congestion_metrics(0, 1, thresholds)
        assert result["congestion_level"] == "Low"

    def test_batch_risk_scoring(self, clean_df):
        thresholds = fit_risk_thresholds(clean_df)
        scored = add_risk_scores_to_df(clean_df, thresholds)
        assert "Congestion_Level" in scored.columns
        assert "Risk_Score" in scored.columns
        assert scored["Risk_Score"].between(0, 100).all()

    def test_metrics_result_keys(self, clean_df):
        thresholds = fit_risk_thresholds(clean_df)
        result = calculate_congestion_metrics(50, 1, thresholds)
        for key in ["predicted_vehicles", "risk_score", "congestion_level", "color", "recommendation"]:
            assert key in result


# ── Test: Evaluate ────────────────────────────────────────────────────────────
class TestEvaluate:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = calculate_metrics(y, y)
        assert metrics["MAE"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["R2"] == pytest.approx(1.0, abs=1e-6)

    def test_metrics_keys(self):
        y = np.array([1, 2, 3, 4, 5])
        yhat = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        metrics = calculate_metrics(y, yhat)
        for key in ["MAE", "RMSE", "R2"]:
            assert key in metrics

    def test_non_negative_mae(self):
        y = np.array([10, 20, 30])
        yhat = np.array([12, 18, 28])
        metrics = calculate_metrics(y, yhat)
        assert metrics["MAE"] >= 0
        assert metrics["RMSE"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
