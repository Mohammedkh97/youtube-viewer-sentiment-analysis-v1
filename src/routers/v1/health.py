# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends
from src.schemas.response import HealthResponse
from app.dependencies import get_prediction_service
from src.services.prediction_service import PredictionService

router = APIRouter(tags=["Health"])


@router.get("/", response_model=HealthResponse)
def health_check(service: PredictionService = Depends(get_prediction_service)):
    status = "healthy" if service.is_ready else "degraded - model not loaded"
    return HealthResponse(status=status, version="1.0.0")
