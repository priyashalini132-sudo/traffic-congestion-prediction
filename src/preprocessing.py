import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame, save_processed: bool = False, output_path: str = None) -> pd.DataFrame:
    """
    Cleans and preprocesses raw traffic dataframe:
    - Standardizes date/time strings to pandas Datetime.
    - Checks and reports missing values (fills/interpolates if any, though dataset is complete).
    - Removes exact duplicate rows.
    - Flags and sanitizes invalid values (e.g., negative vehicles).
    - Sorts the dataset chronologically.
    - Saves clean CSV if requested.
    """
    logger.info("Starting data preprocessing...")
    
    # 1. Copy to avoid modifying original dataframe
    cleaned_df = df.copy()
    
    # 2. Parse DateTime
    try:
        cleaned_df["DateTime"] = pd.to_datetime(cleaned_df["DateTime"])
        logger.info("DateTime column successfully parsed to datetime objects.")
    except Exception as e:
        logger.error(f"Error parsing DateTime column: {e}")
        raise ValueError(f"Failed to parse DateTime values: {e}")
        
    # 3. Check for duplicates
    initial_rows = len(cleaned_df)
    cleaned_df.drop_duplicates(subset=["DateTime", "Junction"], inplace=True)
    duplicate_count = initial_rows - len(cleaned_df)
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate rows (based on DateTime and Junction combination).")
        
    # 4. Handle Missing Values
    null_counts = cleaned_df.isnull().sum()
    logger.info(f"Null value summary before processing:\n{null_counts.to_string()}")
    
    if null_counts.any():
        # Interpolate numerical values chronologically or fill missing values
        logger.warning("Missing values found in the dataset. Interpolating...")
        cleaned_df = cleaned_df.sort_values(by=["Junction", "DateTime"])
        cleaned_df["Vehicles"] = cleaned_df.groupby("Junction")["Vehicles"].transform(
            lambda x: x.interpolate(method="time").ffill().bfill()
        )
        
    # 5. Sanitize anomalous/invalid values
    # Vehicles count should not be negative
    negative_vehicles = (cleaned_df["Vehicles"] < 0).sum()
    if negative_vehicles > 0:
        logger.warning(f"Found {negative_vehicles} negative vehicle count records. Setting them to 0.")
        cleaned_df.loc[cleaned_df["Vehicles"] < 0, "Vehicles"] = 0
        
    # 6. Chronological Sorting
    # Extremely important for time series and chronological train/test splitting
    cleaned_df.sort_values(by=["DateTime", "Junction"], inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Preprocessing completed. Final shape: {cleaned_df.shape}")
    
    # 7. Save Cleaned Data
    if save_processed:
        if output_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(base_dir, "data", "processed", "cleaned_traffic.csv")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved to {output_path}")
        
    return cleaned_df

def generate_preprocessing_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Generate a summary of changes from raw to cleaned data."""
    summary = {
        "raw_shape": raw_df.shape,
        "clean_shape": clean_df.shape,
        "duplicates_removed": len(raw_df) - len(clean_df),
        "missing_values_raw": int(raw_df.isnull().sum().sum()),
        "missing_values_clean": int(clean_df.isnull().sum().sum()),
        "min_date": str(clean_df["DateTime"].min()),
        "max_date": str(clean_df["DateTime"].max()),
        "junctions": [int(j) for j in clean_df["Junction"].unique()]
    }
    return summary
