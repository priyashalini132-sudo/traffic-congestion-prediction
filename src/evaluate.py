from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate core regression evaluation metrics:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R-squared (R2) score
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

def save_model_comparison(comparison_df: pd.DataFrame, output_path: str = None):
    """
    Save the model comparison metrics to a CSV file.
    Default path: reports/model_comparison.csv
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "reports", "model_comparison.csv")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"Saved model comparison table to {output_path}")
