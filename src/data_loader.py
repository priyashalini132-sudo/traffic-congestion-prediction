import os
import pandas as pd
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATASET_URL = "https://raw.githubusercontent.com/ayushabrol13/Traffic-Congestion-Estimation/master/traffic.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset"

def get_default_data_paths():
    """Get standard relative paths for raw data."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    file_path = os.path.join(raw_dir, "traffic.csv")
    return raw_dir, file_path

def download_dataset(destination_path: str):
    """Download the dataset from the raw github url."""
    logger.info(f"Attempting to download dataset from {DATASET_URL}...")
    try:
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        with open(destination_path, "wb") as f:
            f.write(response.content)
            
        logger.info(f"Dataset successfully downloaded and saved to {destination_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False

def load_raw_data(file_path: str = None) -> pd.DataFrame:
    """
    Load raw traffic data.
    If local file is missing, attempts to download it.
    If downloading fails, raises FileNotFoundError with helpful setup instructions.
    """
    raw_dir, default_path = get_default_data_paths()
    if file_path is None:
        file_path = default_path
        
    if not os.path.exists(file_path):
        logger.warning(f"Local dataset not found at {file_path}")
        success = download_dataset(file_path)
        if not success:
            error_msg = (
                f"Dataset not found at '{file_path}' and automated download failed.\n"
                f"Please download 'traffic.csv' manually from Kaggle:\n"
                f"{KAGGLE_URL}\n"
                f"And place it inside the directory: '{raw_dir}'"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset from {file_path}. Shape: {df.shape}")
        
        # Basic validation of expected columns
        expected_cols = ["DateTime", "Junction", "Vehicles", "ID"]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Loaded CSV schema does not match. Missing columns: {missing_cols}. "
                f"Expected columns: {expected_cols}"
            )
            
        return df
    except Exception as e:
        logger.error(f"Error loading dataset at {file_path}: {e}")
        raise

if __name__ == "__main__":
    # Test data loading when executed directly
    try:
        df = load_raw_data()
        print("Success! Dataset head:")
        print(df.head())
    except Exception as e:
        print(f"Loading failed: {e}")
