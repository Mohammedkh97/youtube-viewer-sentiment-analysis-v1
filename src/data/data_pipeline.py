import logging
import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader
from src.data.preprocessor import TextCleanerTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_data_pipeline():
    logger.info("🚀 Starting Data Preprocessing Pipeline...")
    
    raw_data_url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
    output_dir = Path("data/processed")
    output_path = output_dir / "cleaned_data.csv"
    
    # 1. Load Raw Data
    logger.info(f"Downloading raw data from {raw_data_url}...")
    # Note: Using pandas to read directly from URL
    try:
        df = pd.read_csv(raw_data_url)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return
        
    logger.info(f"Raw data loaded. Shape: {df.shape}")
    
    # 2. Structural Cleaning using DataLoader logic
    logger.info("Applying structural cleaning...")
    # Map the columns expected by our system
    if 'clean_comment' not in df.columns and 'comment' in df.columns:
        df = df.rename(columns={'comment': 'clean_comment'})
        
    loader = DataLoader(text_column="clean_comment", target_column="category")
    # Apply mapping and structural cleaning manually since loader expects file path normally
    if 'category' in df.columns and -1 in df['category'].unique():
        df['category'] = df['category'].map({-1: 2, 1: 1, 0: 0})
    
    df_structured = loader._clean_structure(df)
    logger.info(f"Structural cleaning complete. Shape: {df_structured.shape}")
    
    # 3. Text Preprocessing
    logger.info("Applying text cleaning transformer (Lemmatization, Stopwords, etc.)...")
    transformer = TextCleanerTransformer(text_column="clean_comment")
    df_clean = transformer.transform(df_structured)
    
    # 4. Save to Disk
    logger.info(f"Saving cleaned data to {output_path}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    logger.info("✅ Data Preprocessing Pipeline completed successfully!")

if __name__ == "__main__":
    run_data_pipeline()
