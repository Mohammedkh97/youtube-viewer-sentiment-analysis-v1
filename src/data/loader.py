import pandas as pd
from pathlib import Path
from typing import Union

class DataLoader:
    """Class responsible for loading and initial structural cleaning of datasets."""
    def __init__(self, text_column: str = 'clean_comment', target_column: str = 'category'):
        self.text_column = text_column
        self.target_column = target_column
        
    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Loads data from CSV and applies structural cleaning."""
        df = pd.read_csv(filepath)
        
        # Initial mapping if necessary based on your dataset format
        if self.target_column in df.columns:
            # Check if it needs the -1 -> 2 mapping
            if -1 in df[self.target_column].unique():
                df[self.target_column] = df[self.target_column].map({-1: 2, 1: 1, 0: 0})
        
        return self._clean_structure(df)
        
    def _clean_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops NaNs, duplicates, and structurally empty strings."""
        df_cleaned = df.copy()
        
        if self.text_column in df_cleaned.columns:
            df_cleaned.dropna(subset=[self.text_column], inplace=True)
            df_cleaned.drop_duplicates(inplace=True)
            df_cleaned = df_cleaned[df_cleaned[self.text_column].astype(str).str.strip() != ""]
            
        if self.target_column in df_cleaned.columns:
            df_cleaned.dropna(subset=[self.target_column], inplace=True)
            
        return df_cleaned
