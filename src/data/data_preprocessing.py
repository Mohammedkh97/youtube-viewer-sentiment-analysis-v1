"""
DVC Stage 2: Data Preprocessing
Reads raw train/test CSVs, applies text cleaning, and saves processed data.
"""
import logging
import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader
from src.data.preprocessor import TextCleanerTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_data_preprocessing():
    logger.info("🚀 Stage 2: Data Preprocessing...")

    raw_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(text_column="clean_comment", target_column="category")
    transformer = TextCleanerTransformer(text_column="clean_comment")

    for split in ["train", "test"]:
        input_path = raw_dir / f"{split}.csv"
        logger.info(f"Processing {input_path}...")

        df = pd.read_csv(input_path)
        df_clean = loader._clean_structure(df)
        df_clean = transformer.transform(df_clean)

        output_path = output_dir / f"{split}_processed.csv"
        df_clean.to_csv(output_path, index=False)
        logger.info(f"  Saved {output_path} | Shape: {df_clean.shape}")

    logger.info("✅ Data Preprocessing complete.")

if __name__ == "__main__":
    run_data_preprocessing()
