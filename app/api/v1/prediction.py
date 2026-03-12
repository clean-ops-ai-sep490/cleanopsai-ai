from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_service import model_service

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, summary="Run model inference")
def predict(request: PredictionRequest) -> PredictionResponse:
    """Accept input features and return the model prediction."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    result = model_service.predict(request.features)
    return PredictionResponse(**result)
