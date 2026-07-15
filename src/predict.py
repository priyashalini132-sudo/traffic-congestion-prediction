import os
import joblib
import pandas as pd
import numpy as np
import logging
from src.feature_engineering import build_features
from src.risk_scoring import calculate_congestion_metrics

logger = logging.getLogger(__name__)

class TrafficPredictor:
    """Class to manage loading models and running inference on user inputs."""
    
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "models")
            
        self.model_path = os.path.join(model_dir, "best_model.pkl")
        self.prep_path = os.path.join(model_dir, "preprocessor.pkl")
        
        self.model = None
        self.preprocessor = None
        self.loaded = False
        
    def load_model(self):
        """Loads serialized model and preprocessor files."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.prep_path):
            error_msg = (
                f"Model artifacts not found. Missing:\n"
                f"  - Model: {self.model_path} ({os.path.exists(self.model_path)})\n"
                f"  - Preprocessor: {self.prep_path} ({os.path.exists(self.prep_path)})\n"
                f"Please execute 'python -m src.train' to train models and generate these artifacts."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.prep_path)
            self.loaded = True
            logger.info("Successfully loaded model and preprocessor.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise RuntimeError(f"Failed to load machine learning model: {e}")
            
    def predict(self, dt: pd.Timestamp, junction: int) -> dict:
        """
        Run inference for a single DateTime and Junction pair:
        1. Form single-row DataFrame.
        2. Perform feature engineering (temporal, cyclical, historical profiles).
        3. One-hot encode Junction.
        4. Standard-scale features.
        5. Execute model prediction.
        6. Compute risk scores and recommendations.
        """
        if not self.loaded:
            self.load_model()
            
        # 1. Create raw row
        raw_row = pd.DataFrame([{
            "DateTime": pd.to_datetime(dt),
            "Junction": int(junction)
        }])
        
        # 2. Build features
        # Extract profiles from preprocessor
        profiles = self.preprocessor["historical_profiles"]
        featured_row = build_features(raw_row, profiles)
        
        # 3. Create One-Hot encoded Junction indicators
        for j in [1, 2, 3, 4]:
            featured_row[f"Junction_{j}"] = 1 if junction == j else 0
            
        # 4. Extract standard features list
        feature_cols = self.preprocessor["feature_names"]
        X_raw = featured_row[feature_cols].copy()
        X_raw.fillna(0.0, inplace=True)
        
        # 5. Apply scale
        cols_to_scale = ["Hour", "Day", "DayOfWeek", "Month", "Year", "Hist_Traffic_J_D_H", "Hist_Traffic_J_H"]
        scaler = self.preprocessor["scaler"]
        
        X_scaled_part = scaler.transform(X_raw[cols_to_scale])
        X_scaled = X_raw.copy()
        X_scaled[cols_to_scale] = X_scaled_part
        
        # 6. Predict
        X = X_scaled.values
        predicted_vehicles = float(self.model.predict(X)[0])
        
        # Enforce non-negative output
        predicted_vehicles = max(0.0, predicted_vehicles)
        
        # 7. Apply risk scoring
        risk_thresholds = self.preprocessor["risk_thresholds"]
        metrics = calculate_congestion_metrics(predicted_vehicles, junction, risk_thresholds)
        
        return metrics

if __name__ == "__main__":
    # Test predictor load and prediction
    import pandas as pd
    predictor = TrafficPredictor()
    try:
        predictor.load_model()
        test_dt = pd.Timestamp("2026-10-12 08:00:00")
        test_junction = 1
        res = predictor.predict(test_dt, test_junction)
        print(f"Prediction for Junction {test_junction} on {test_dt}:")
        print(res)
    except Exception as e:
        print(f"Error testing predictor: {e}")
