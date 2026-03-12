import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api.v1 import api_router
from app.core.config import settings
from app.services.model_service import model_service

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    loaded = model_service.load()
    if loaded:
        logger.info("Model loaded successfully on startup")
    else:
        logger.warning(
            "No model found on startup — /predict will return 503 until a model is placed in %s",
            settings.model_path,
        )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI service for CleanOpsAI — provides model inference via a REST API.",
        lifespan=lifespan,
    )

    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()
