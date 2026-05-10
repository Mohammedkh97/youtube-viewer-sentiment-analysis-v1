"""
DVC Stage 1: Data Ingestion
Downloads the raw dataset and splits it into train/test sets.
"""
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_data_ingestion():
    logger.info("🚀 Stage 1: Data Ingestion...")

    raw_data_url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download raw data
    logger.info(f"Downloading raw data from {raw_data_url}...")
    try:
        df = pd.read_csv(raw_data_url)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    logger.info(f"Raw data loaded. Shape: {df.shape}")

    # 2. Rename columns to match our schema
    if "clean_comment" not in df.columns and "comment" in df.columns:
        df = df.rename(columns={"comment": "clean_comment"})

    # 3. Map category labels: -1 -> 2 (positive)
    if "category" in df.columns and -1 in df["category"].unique():
        df["category"] = df["category"].map({-1: 2, 1: 1, 0: 0})

    # 4. Train/Test split
    test_size = settings.ml_params.get("data_ingestion", {}).get("test_size", 0.2)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["category"]
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"✅ Data Ingestion complete. Train: {train_df.shape}, Test: {test_df.shape}")

if __name__ == "__main__":
    run_data_ingestion()
