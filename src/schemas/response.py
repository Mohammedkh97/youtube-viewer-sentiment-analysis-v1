from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    comment: str
    sentiment: str
    confidence: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    version: str
