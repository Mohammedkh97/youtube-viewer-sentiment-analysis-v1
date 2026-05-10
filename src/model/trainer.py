import os

# pyrefly: ignore [missing-import]
import mlflow

# pyrefly: ignore [missing-import]
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

from src.data.preprocessor import VectorizerFactory
from src.model.tuner import ModelFactory
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ImbalanceHandlerFactory:
    @staticmethod
    def get(name: str, random_state=42):
        name = name.lower()
        handlers = {
            "oversampling": SMOTE(random_state=random_state),
            "adasyn": ADASYN(random_state=random_state),
            "undersampling": RandomUnderSampler(random_state=random_state),
            "smote_enn": SMOTEENN(random_state=random_state),
        }
        return handlers.get(name)


class Trainer:
    """
    Orchestrates the full training pipeline:
        encode -> split -> vectorize -> (optionally resample) -> tune/train -> evaluate -> MLflow log
    """

    def __init__(
        self,
        experiment_name="YouTube_Sentiment_Pipeline",
        text_column="clean_comment",
        target_column="category",
        test_size=0.2,
        random_state=42,
    ):
        self.text_column = text_column
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

    def _cast_features(self, X, model_name):
        if model_name.lower() in ["lightgbm", "xgboost"]:
            return X.astype("float32")
        return X

    def train(
        self,
        df,
        param_grids: dict,
        vectorizer_name="tfidf",
        model_name="random_forest",
        imbalance_method="class_weights",
        ngram_range=(1, 3),
        max_features=10000,
        search_strategy: str = "manual",
        cv_strategy: str = "stratified_kfold",
        n_splits: int = 5,
        n_iter: int = 20,
        n_trials: int = 30,
        scoring: str = "f1_macro",
    ):
        df = df.copy()
        X = df[self.text_column]
        y = self.label_encoder.fit_transform(df[self.target_column])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        vectorizer = VectorizerFactory.get(
            vectorizer_name, ngram_range=ngram_range, max_features=max_features
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        if imbalance_method != "class_weights":
            sampler = ImbalanceHandlerFactory.get(
                imbalance_method, random_state=self.random_state
            )
            if sampler:
                X_train_vec, y_train = sampler.fit_resample(X_train_vec, y_train)

        X_train_vec = self._cast_features(X_train_vec, model_name)
        X_test_vec = self._cast_features(X_test_vec, model_name)

        model = ModelFactory.get(
            name=model_name,
            param_grids=param_grids,
            random_state=self.random_state,
            search_strategy=search_strategy,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            n_iter=n_iter,
            n_trials=n_trials,
            X_train=X_train_vec,
            y_train=y_train,
            scoring=scoring,
            verbose=0,
        )

        if imbalance_method == "class_weights" and hasattr(model, "class_weight"):
            model.set_params(class_weight="balanced")

        with mlflow.start_run():
            mlflow.log_param("vectorizer", vectorizer_name)
            mlflow.log_param("model", model_name)
            mlflow.log_param("imbalance_method", imbalance_method)
            mlflow.log_param("search_strategy", search_strategy)

            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

            logger.info(f"Accuracy: {acc:.4f}")

            # Save artifacts locally and in MLFlow
            os.makedirs(settings.artifacts_dir, exist_ok=True)
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"{model_name} | {search_strategy}")
            cm_path = settings.artifacts_dir / f"cm_{model_name}_{search_strategy}.png"
            plt.savefig(cm_path)
            plt.close()

            mlflow.log_artifact(cm_path)
            mlflow.sklearn.log_model(model, artifact_path="model")

        return model, vectorizer, self.label_encoder
