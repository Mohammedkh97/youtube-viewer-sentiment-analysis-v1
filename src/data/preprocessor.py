from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from typing import Optional

from src.utils.text_utils import TextCleanerStrategy, DefaultTextCleaner

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for text cleaning."""
    def __init__(self, text_column: str = 'clean_comment', strategy: Optional[TextCleanerStrategy] = None):
        self.text_column = text_column
        self.strategy = strategy or DefaultTextCleaner()
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = X.copy()
        
        if self.text_column not in X_processed.columns:
            return X_processed
            
        X_processed[self.text_column] = X_processed[self.text_column].apply(self.strategy.clean)
        return X_processed

class VectorizerFactory:
    """Factory to instantiate different vectorizers."""
    @staticmethod
    def get(name: str, ngram_range=(1, 1), max_features=8000):
        name = name.lower()
        if name == "tfidf":
            return TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        elif name == "count":
            return CountVectorizer(ngram_range=ngram_range, max_features=max_features)
        else:
            raise ValueError(f"Unknown vectorizer: {name}")
