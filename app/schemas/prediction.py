from pydantic import BaseModel, Field
from typing import Any


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., description="Input feature values for prediction")


class PredictionResponse(BaseModel):
    prediction: Any = Field(..., description="Model prediction result")
    probability: float | None = Field(None, description="Prediction confidence (0-1)")


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
