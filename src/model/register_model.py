"""
DVC Stage 5: Model Registration
Reads experiment_info.json and registers the model + metrics in MLflow.
"""

import json
import joblib
import logging

# pyrefly: ignore [missing-import]
import mlflow

# pyrefly: ignore [missing-import]
import mlflow.sklearn
from pathlib import Path

from src.config.settings import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_model_registration():
    logger.info("🚀 Stage 5: Model Registration...")

    # 1. Load experiment info
    experiment_path = Path("metrics/experiment_info.json")
    with open(experiment_path, "r") as f:
        experiment_info = json.load(f)

    metrics = experiment_info["metrics"]
    model_class = experiment_info["model_class"]

    # 2. Load model artifact
    model = joblib.load(settings.model_path)

    # 3. Register in MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.app_name)

    with mlflow.start_run(run_name=f"DVC_Pipeline_{model_class}"):
        # Log params
        ml_params = settings.ml_params
        model_cfg = ml_params.get("model_training", {})
        mlflow.log_param("model_class", model_class)
        mlflow.log_param("max_features", model_cfg.get("max_features"))
        mlflow.log_param("ngram_range", str(model_cfg.get("ngram_range")))
        mlflow.log_param("n_estimators", model_cfg.get("n_estimators"))

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Log the confusion matrix image if it exists
        for cm_file in settings.artifacts_dir.glob("cm_*.png"):
            mlflow.log_artifact(str(cm_file))

        # Log experiment_info.json itself
        mlflow.log_artifact(str(experiment_path))

    logger.info(
        f"✅ Model Registration complete. Registered to MLflow: {settings.mlflow_tracking_uri}"
    )


if __name__ == "__main__":
    run_model_registration()
