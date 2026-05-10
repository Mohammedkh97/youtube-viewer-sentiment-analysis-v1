import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Protocol

class TextCleanerStrategy(Protocol):
    """Strategy pattern interface for text cleaning."""
    def clean(self, text: str) -> str:
        pass

class DefaultTextCleaner(TextCleanerStrategy):
    """Default implementation of text cleaning using regex and NLTK."""
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
            
    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
            
        text = text.lower().strip()
        text = re.sub(r'\n', ' ', text)
        
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        text = re.sub(url_pattern, '', text)
        text = re.sub(r'[^A-Za-z0-9\s!?.,]', '', text)
        
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
