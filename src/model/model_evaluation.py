"""
DVC Stage 4: Model Evaluation
Loads the trained model + vectorizer, evaluates on the test set,
and saves metrics and confusion matrix to a JSON file.
"""
import json
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_model_evaluation():
    logger.info("🚀 Stage 4: Model Evaluation...")

    # 1. Load artifacts
    model = joblib.load(settings.model_path)
    vectorizer = joblib.load(settings.preprocessor_path)
    label_encoder = joblib.load(settings.artifacts_dir / "label_encoder.joblib")

    # 2. Load test data
    test_path = Path("data/processed/test_processed.csv")
    df_test = pd.read_csv(test_path)
    df_test = df_test.dropna(subset=["clean_comment"])
    logger.info(f"Test data loaded. Shape: {df_test.shape}")

    X_test = df_test["clean_comment"]
    y_test = label_encoder.transform(df_test["category"])

    X_test_vec = vectorizer.transform(X_test)

    model_name = type(model).__name__
    if model_name in ["LGBMClassifier", "XGBClassifier"]:
        X_test_vec = X_test_vec.astype("float32")

    # 3. Predict & Compute Metrics
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }

    report = classification_report(y_test, y_pred, output_dict=True)

    experiment_info = {
        "model_class": model_name,
        "metrics": metrics,
        "classification_report": report,
    }

    # 4. Generate and save Confusion Matrix Plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    cm_path = settings.artifacts_dir / f"cm_{model_name}.png"
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    # 5. Save metrics
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / "experiment_info.json"

    with open(output_path, "w") as f:
        json.dump(experiment_info, f, indent=2, default=str)

    logger.info(f"  Accuracy:        {metrics['accuracy']}")
    logger.info(f"  F1 (macro):      {metrics['f1_macro']}")
    logger.info(f"  Precision:       {metrics['precision_macro']}")
    logger.info(f"  Recall:          {metrics['recall_macro']}")
    logger.info(f"✅ Model Evaluation complete. Metrics saved to {output_path}")

if __name__ == "__main__":
    run_model_evaluation()
