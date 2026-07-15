import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from DateTime column:
    - Hour, Day, Month, Year
    - Day of week (0=Mon, 6=Sun)
    - Weekend indicator (1=Sat/Sun, 0=Weekday)
    - Is_Rush_Hour indicator (7-9 AM, 5-7 PM on weekdays)
    - Time_Of_Day categorizer
    """
    df = df.copy()
    
    # Extract components
    df["Hour"] = df["DateTime"].dt.hour
    df["Day"] = df["DateTime"].dt.day
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Month"] = df["DateTime"].dt.month
    df["Year"] = df["DateTime"].dt.year
    
    # Weekend indicator
    df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
    
    # Rush Hour indicator (commute hours 7-9 and 17-19 on weekdays)
    df["IsRushHour"] = ((df["Hour"].isin([7, 8, 9, 17, 18])) & (df["IsWeekend"] == 0)).astype(int)
    
    # Time of Day (numeric representation of blocks)
    # 0: Night (22-5), 1: Morning (6-11), 2: Afternoon (12-16), 3: Evening (17-21)
    def categorize_time_of_day(hour):
        if 6 <= hour < 12:
            return 1  # Morning
        elif 12 <= hour < 17:
            return 2  # Afternoon
        elif 17 <= hour < 22:
            return 3  # Evening
        else:
            return 0  # Night
            
    df["TimeOfDay"] = df["Hour"].apply(categorize_time_of_day)
    
    return df

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode cyclical time features using sine and cosine transformations:
    - Hour (24-hour cycle)
    - Day of week (7-day cycle)
    - Month (12-month cycle)
    """
    df = df.copy()
    
    # Hour cyclical features
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    
    # Day of week cyclical features
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7.0)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7.0)
    
    # Month cyclical features
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    
    return df

def fit_historical_profiles(train_df: pd.DataFrame) -> dict:
    """
    Calculate and extract historical mean congestion patterns per junction
    from the training set. This acts as a robust feature mapping for unseen/future 
    data inputs, avoiding standard lag-feature data leakage.
    
    Returns a dictionary of mappings.
    """
    logger.info("Fitting historical congestion profiles on training set...")
    
    # Ensure temporal features are present
    if "Hour" not in train_df.columns:
        train_df = add_temporal_features(train_df)
        
    # Map 1: Junction-specific baseline average traffic
    junction_mean = train_df.groupby("Junction")["Vehicles"].mean().to_dict()
    
    # Map 2: Junction x Hour baseline traffic
    j_hour_mean = train_df.groupby(["Junction", "Hour"])["Vehicles"].mean().to_dict()
    
    # Map 3: Junction x DayOfWeek x Hour baseline traffic (Highly predictive)
    j_dow_hour_mean = train_df.groupby(["Junction", "DayOfWeek", "Hour"])["Vehicles"].mean().to_dict()
    
    return {
        "overall_mean": float(train_df["Vehicles"].mean()),
        "junction_mean": junction_mean,
        "j_hour_mean": j_hour_mean,
        "j_dow_hour_mean": j_dow_hour_mean
    }

def apply_historical_profiles(df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    """
    Map historical traffic volume averages to a dataset using fitted profiles.
    Fills missing matches using broader historical averages.
    """
    df = df.copy()
    
    # Extract profiles
    overall_mean = profiles["overall_mean"]
    junction_mean = profiles["junction_mean"]
    j_hour_mean = profiles["j_hour_mean"]
    j_dow_hour_mean = profiles["j_dow_hour_mean"]
    
    # 1. Map Junction x DayOfWeek x Hour
    def map_j_dow_hour(row):
        key = (int(row["Junction"]), int(row["DayOfWeek"]), int(row["Hour"]))
        if key in j_dow_hour_mean:
            return j_dow_hour_mean[key]
        
        # Fallback 1: Junction x Hour
        key_fallback = (int(row["Junction"]), int(row["Hour"]))
        if key_fallback in j_hour_mean:
            return j_hour_mean[key_fallback]
            
        # Fallback 2: Junction
        j_key = int(row["Junction"])
        if j_key in junction_mean:
            return junction_mean[j_key]
            
        # Fallback 3: Overall mean
        return overall_mean
        
    df["Hist_Traffic_J_D_H"] = df.apply(map_j_dow_hour, axis=1)
    
    # 2. Map Junction x Hour
    def map_j_hour(row):
        key = (int(row["Junction"]), int(row["Hour"]))
        if key in j_hour_mean:
            return j_hour_mean[key]
        
        j_key = int(row["Junction"])
        if j_key in junction_mean:
            return junction_mean[j_key]
            
        return overall_mean
        
    df["Hist_Traffic_J_H"] = df.apply(map_j_hour, axis=1)
    
    return df

def build_features(df: pd.DataFrame, profiles: dict = None) -> pd.DataFrame:
    """
    Execute full feature engineering pipeline:
    - Temporal features
    - Cyclical features
    - Mapping of historical profiles if provided
    """
    df = add_temporal_features(df)
    df = add_cyclical_features(df)
    
    if profiles is not None:
        df = apply_historical_profiles(df, profiles)
        
    return df
