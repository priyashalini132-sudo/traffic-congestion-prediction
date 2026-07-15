import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def fit_risk_thresholds(train_df: pd.DataFrame) -> dict:
    """
    Calculate junction-specific quantiles and maximum volumes for risk scoring.
    We use the 33.3rd and 66.6th percentiles to split traffic volume into:
    - Low Risk
    - Medium Risk
    - High Risk
    
    Returns a dictionary of thresholds and capacity limits per junction.
    """
    logger.info("Fitting junction-specific congestion risk thresholds on training set...")
    
    thresholds = {}
    unique_junctions = train_df["Junction"].unique()
    
    for junction in unique_junctions:
        j_data = train_df[train_df["Junction"] == junction]["Vehicles"]
        
        # Calculate thresholds
        low_thresh = float(np.quantile(j_data, 0.33))
        high_thresh = float(np.quantile(j_data, 0.66))
        max_val = float(j_data.max())
        
        # Prevent division by zero
        if max_val == 0:
            max_val = 1.0
            
        thresholds[str(junction)] = {
            "low_threshold": low_thresh,
            "high_threshold": high_thresh,
            "max_volume": max_val
        }
        
    # Global fallback if a junction isn't found
    global_vehicles = train_df["Vehicles"]
    thresholds["global"] = {
        "low_threshold": float(np.quantile(global_vehicles, 0.33)),
        "high_threshold": float(np.quantile(global_vehicles, 0.66)),
        "max_volume": float(global_vehicles.max()) if global_vehicles.max() > 0 else 1.0
    }
    
    return thresholds

def calculate_congestion_metrics(vehicles: float, junction: int, thresholds: dict) -> dict:
    """
    Calculate the risk score, congestion level, and recommendation
    for a given traffic volume and junction.
    
    - Risk Score: 0 to 100, representing percentage of maximum observed volume for that junction.
    - Congestion Level: Low, Medium, High.
    - Recommendation: Travel advisory.
    """
    j_key = str(junction)
    if j_key in thresholds:
        thresh = thresholds[j_key]
    else:
        thresh = thresholds["global"]
        
    low_t = thresh["low_threshold"]
    high_t = thresh["high_threshold"]
    max_v = thresh["max_volume"]
    
    # 1. Congestion Risk Score (normalized 0-100)
    risk_score = min(100.0, max(0.0, (vehicles / max_v) * 100.0))
    
    # 2. Congestion Level
    if vehicles <= low_t:
        level = "Low"
        color = "#10B981"  # Emerald green
        recommendation = "Traffic conditions are currently manageable."
    elif vehicles <= high_t:
        level = "Medium"
        color = "#F59E0B"  # Amber/orange
        recommendation = "Moderate congestion expected. Consider alternative travel times."
    else:
        level = "High"
        color = "#EF4444"  # Red
        recommendation = "High congestion risk. Avoid peak periods or use an alternative route if possible."
        
    return {
        "predicted_vehicles": float(round(vehicles, 2)),
        "risk_score": float(round(risk_score, 1)),
        "congestion_level": level,
        "color": color,
        "recommendation": recommendation
    }

def add_risk_scores_to_df(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Batch apply risk scoring to a dataframe containing predictions or actuals.
    Adds 'Risk_Score', 'Congestion_Level', 'Color', and 'Recommendation' columns.
    """
    df = df.copy()
    
    levels = []
    scores = []
    colors = []
    recs = []
    
    for _, row in df.iterrows():
        vehicles = row["Vehicles"]
        junction = row["Junction"]
        metrics = calculate_congestion_metrics(vehicles, junction, thresholds)
        
        levels.append(metrics["congestion_level"])
        scores.append(metrics["risk_score"])
        colors.append(metrics["color"])
        recs.append(metrics["recommendation"])
        
    df["Congestion_Level"] = levels
    df["Risk_Score"] = scores
    df["Color"] = colors
    df["Recommendation"] = recs
    
    return df
