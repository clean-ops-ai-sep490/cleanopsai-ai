from fastapi import APIRouter

from app.core.config import settings
from app.schemas.prediction import HealthResponse
from app.services.model_service import model_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    """Return application health and model status."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        model_loaded=model_service.is_loaded,
    )
