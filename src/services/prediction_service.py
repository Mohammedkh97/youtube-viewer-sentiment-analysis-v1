import joblib
import pandas as pd
import numpy as np
import logging
from src.config.settings import settings
from src.data.preprocessor import TextCleanerTransformer

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.transformer = TextCleanerTransformer(text_column="comment")
        self.is_ready = False

    def load_artifacts(self):
        try:
            logger.info("Loading model artifacts...")
            if not settings.model_path.exists():
                logger.warning(f"Model not found at {settings.model_path}. Service will not be ready.")
                return
                
            self.model = joblib.load(settings.model_path)
            self.vectorizer = joblib.load(settings.preprocessor_path)
            self.label_encoder = joblib.load(settings.artifacts_dir / "label_encoder.joblib")
            self.is_ready = True
            logger.info("Artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            self.is_ready = False

    def predict(self, texts: list[str]) -> list[dict]:
        if not self.is_ready:
            raise RuntimeError("Model artifacts are not loaded.")

        df = pd.DataFrame({"comment": texts})
        df_clean = self.transformer.transform(df)
        
        X_vec = self.vectorizer.transform(df_clean["comment"])
        if type(self.model).__name__ in ["LGBMClassifier", "XGBClassifier"]:
            X_vec = X_vec.astype("float32")
            
        predictions = self.model.predict(X_vec)
        probabilities = self.model.predict_proba(X_vec)
        
        results = []
        for i in range(len(texts)):
            pred_idx = predictions[i]
            confidence = float(np.max(probabilities[i]))
            sentiment = self.label_encoder.inverse_transform([pred_idx])[0]
            
            # In data_ingestion.py, the mapping was: -1(negative)->2, 0(neutral)->0, 1(positive)->1
            sentiment_map = {0: "neutral", 1: "positive", 2: "negative"}
            sentiment_label = sentiment_map.get(sentiment, str(sentiment))
            
            results.append({
                "comment": texts[i],
                "sentiment": sentiment_label,
                "confidence": confidence
            })
            
        return results
