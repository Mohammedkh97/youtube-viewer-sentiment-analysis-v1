import yaml
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Any, Dict

def load_yaml_params(filepath: str = "params.yaml") -> Dict[str, Any]:
    """Loads parameters from the yaml file."""
    try:
        with open(filepath, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables and params.yaml.
    Environment variables take precedence over default values.
    """
    # Application Config
    app_name: str = "YouTube Viewer Sentiment API"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    
    # MLflow Config
    mlflow_tracking_uri: str = "https://mlflow-dashboard.duckdns.org"
    
    # Paths
    artifacts_dir: Path = Path("artifacts")
    model_path: Path = Path("artifacts/model.joblib")
    preprocessor_path: Path = Path("artifacts/preprocessor.joblib")
    
    # ML Parameters
    ml_params: Dict[str, Any] = Field(default_factory=load_yaml_params)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Global settings instance
settings = Settings()
