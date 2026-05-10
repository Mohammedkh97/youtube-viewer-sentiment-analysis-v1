from src.services.prediction_service import PredictionService

# Singleton pattern for the service
_prediction_service = PredictionService()

def get_prediction_service() -> PredictionService:
    return _prediction_service
