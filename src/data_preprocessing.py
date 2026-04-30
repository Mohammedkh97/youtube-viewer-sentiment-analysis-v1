import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextCleaner:
    """Class to clean and preprocess raw text comments."""
    def __init__(self):
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        self.lemmatizer = WordNetLemmatizer()
        
    def _ensure_nltk_resources(self):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove trailing and leading whitespaces
        text = text.strip()
        
        # Remove newline characters
        text = re.sub(r'\n', ' ', text)
        
        # Remove URLs
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        text = re.sub(url_pattern, '', text)
        
        # Remove non-alphanumeric characters, except punctuation
        text = re.sub(r'[^A-Za-z0-9\s!?.,]', '', text)
        
        # Remove stopwords
        text = ' '.join(word for word in text.split() if word not in self.stop_words)
        
        # Lemmatize words
        text = ' '.join(self.lemmatizer.lemmatize(word) for word in text.split())
        
        return text

class DataPreprocessor:
    """Class to handle DataFrame level data preprocessing."""
    def __init__(self, text_column='clean_comment'):
        self.text_column = text_column
        self.text_cleaner = TextCleaner()
        
    def fit(self, df, y=None):
        return self
        
    def transform(self, df):
        """Executes the full preprocessing pipeline on the input DataFrame."""
        df_processed = df.copy()
        
        if self.text_column not in df_processed.columns:
            return df_processed
            
        # 1. Drop missing values based on text column
        df_processed.dropna(subset=[self.text_column], inplace=True)
        
        # 2. Drop duplicates
        df_processed.drop_duplicates(inplace=True)
        
        # 3. Strip initial empty records before further processing
        df_processed = df_processed[df_processed[self.text_column].str.strip() != ""]
        
        # 4. Apply text cleaning
        df_processed[self.text_column] = df_processed[self.text_column].apply(self.text_cleaner.clean_text)
        
        # 5. Remove any empty strings formed after cleaning
        df_processed = df_processed[df_processed[self.text_column].str.strip() != ""]
        
        return df_processed
