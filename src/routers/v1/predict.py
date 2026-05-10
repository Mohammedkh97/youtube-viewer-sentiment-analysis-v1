# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException
from src.schemas.request import PredictionRequest, BatchPredictionRequest
from src.schemas.response import PredictionResponse, BatchPredictionResponse
from app.dependencies import get_prediction_service
from src.services.prediction_service import PredictionService

router = APIRouter(tags=["Predict"])


@router.post("/", response_model=PredictionResponse)
def predict_sentiment(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    try:
        result = service.predict([request.comment])[0]
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch_sentiment(
    request: BatchPredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    try:
        results = service.predict(request.comments)
        return BatchPredictionResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
