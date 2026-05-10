import joblib
import logging
from pathlib import Path
from src.config.settings import settings
from src.data.loader import DataLoader
from src.data.preprocessor import TextCleanerTransformer
from src.model.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_training_pipeline():
    logger.info("🚀 Starting Training Pipeline...")
    
    # 1. Load Data
    logger.info("Loading raw data...")
    # Assuming the raw data exists in data/processed/cleaned_data.csv as per notebook 4
    # In a real DVC setup, this would be data/raw/raw_comments.csv
    data_path = Path("data/processed/cleaned_data.csv")
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}. Please ensure it exists.")
        return
        
    loader = DataLoader(text_column="clean_comment", target_column="category")
    df = loader.load_data(data_path)
    logger.info(f"Data loaded and structurally cleaned. Shape: {df.shape}")
    
    # 2. Text Preprocessing
    logger.info("Applying text cleaning transformer...")
    transformer = TextCleanerTransformer(text_column="clean_comment")
    df_clean = transformer.transform(df)
    
    # 3. Model Training
    logger.info("Initializing trainer...")
    trainer = Trainer(
        experiment_name=settings.app_name,
        text_column="clean_comment",
        target_column="category",
        test_size=settings.ml_params.get("data_ingestion", {}).get("test_size", 0.2)
    )
    
    # Read hyperparams from params.yaml via settings
    param_grids = settings.ml_params
    
    # Example: Run a bayesian tuning on xgboost
    tuning_params = param_grids.get("bayesian_tuning", {})
    model_name = tuning_params.get("model_name", "xgboost")
    search_strategy = tuning_params.get("search_strategy", "bayesian")
    cv_strategy = tuning_params.get("cv_strategy", "stratified_kfold")
    n_splits = tuning_params.get("n_splits", 5)
    n_trials = tuning_params.get("n_trials", 30)
    
    logger.info(f"Starting model training for {model_name} with {search_strategy} search...")
    model, vectorizer, label_encoder = trainer.train(
        df=df_clean,
        param_grids=param_grids,
        model_name=model_name,
        search_strategy=search_strategy,
        cv_strategy=cv_strategy,
        n_splits=n_splits,
        n_trials=n_trials
    )
    
    # 4. Save Artifacts
    logger.info("Saving artifacts...")
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, settings.model_path)
    joblib.dump(vectorizer, settings.preprocessor_path)
    joblib.dump(label_encoder, settings.artifacts_dir / "label_encoder.joblib")
    
    logger.info(f"✅ Pipeline completed successfully. Artifacts saved in {settings.artifacts_dir}")

if __name__ == "__main__":
    run_training_pipeline()
