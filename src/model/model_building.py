"""
DVC Stage 3: Model Building
Reads processed training data, tunes hyperparameters, trains the model,
and saves the model + vectorizer artifacts.
"""
import joblib
import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.data.preprocessor import VectorizerFactory
from src.model.tuner import ModelFactory
from src.model.trainer import ImbalanceHandlerFactory
from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_model_building():
    logger.info("🚀 Stage 3: Model Building...")

    # 1. Load processed training data
    train_path = Path("data/processed/train_processed.csv")
    df = pd.read_csv(train_path)
    logger.info(f"Training data loaded. Shape: {df.shape}")

    text_column = "clean_comment"
    target_column = "category"

    X = df[text_column]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_column])

    # 2. Vectorize
    ml_params = settings.ml_params
    model_cfg = ml_params.get("model_training", {})
    ngram_range = tuple(model_cfg.get("ngram_range", [1, 3]))
    max_features = model_cfg.get("max_features", 10000)

    vectorizer_name = model_cfg.get("vectorizer", "tfidf")
    vectorizer = VectorizerFactory.get(vectorizer_name, ngram_range=ngram_range, max_features=max_features)
    X_vec = vectorizer.fit_transform(X)

    # 3. Handle Imbalanced Data
    imbalance_method = model_cfg.get("imbalance_method", "class_weights")
    if imbalance_method != "class_weights":
        sampler = ImbalanceHandlerFactory.get(imbalance_method, random_state=42)
        if sampler:
            logger.info(f"Applying imbalance handler: {imbalance_method}")
            X_vec, y = sampler.fit_resample(X_vec, y)
            logger.info(f"Resampled data shape: X={X_vec.shape}, y={len(y)}")
        else:
            logger.warning(f"Unknown imbalance method '{imbalance_method}', skipping.")

    # 4. Tune & Build Model
    tuning_cfg = ml_params.get("bayesian_tuning", {})
    model_name = tuning_cfg.get("model_name", "xgboost")
    search_strategy = tuning_cfg.get("search_strategy", "bayesian")
    cv_strategy = tuning_cfg.get("cv_strategy", "stratified_kfold")
    n_splits = tuning_cfg.get("n_splits", 5)
    n_trials = tuning_cfg.get("n_trials", 30)

    logger.info(f"Building model: {model_name} | strategy: {search_strategy}")

    # Cast for tree-based models
    if model_name.lower() in ["lightgbm", "xgboost"]:
        X_vec = X_vec.astype("float32")

    model = ModelFactory.get(
        name=model_name,
        param_grids=ml_params,
        random_state=42,
        search_strategy=search_strategy,
        cv_strategy=cv_strategy,
        n_splits=n_splits,
        n_trials=n_trials,
        X_train=X_vec,
        y_train=y,
        scoring="f1_macro",
        verbose=0,
    )

    model.fit(X_vec, y)
    logger.info("Model trained successfully.")

    # 4. Save artifacts
    artifacts_dir = settings.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, settings.model_path)
    joblib.dump(vectorizer, settings.preprocessor_path)
    joblib.dump(label_encoder, artifacts_dir / "label_encoder.joblib")

    logger.info(f"✅ Model Building complete. Artifacts saved to {artifacts_dir}")

if __name__ == "__main__":
    run_model_building()
