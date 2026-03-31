from typing import List

from pydantic import BaseModel, Field


class AIRequest(BaseModel):
    image_urls: List[str]
    validation_type: str = "all_required"
    required_objects: List[str]
    min_confidence: float = 0.25


class QuickTestRequest(BaseModel):
    image_url: str
    min_confidence: float = 0.25


class SanitationScoreRequest(BaseModel):
    before_img: str = Field(..., description="Before cleaning image URL")
    after_img: str = Field(..., description="After cleaning image URL")


class SanitationSnapshot(BaseModel):
    stains: int
    coverage: float


class SanitationScoreResponse(BaseModel):
    confidence_score: float
    before: SanitationSnapshot
    after: SanitationSnapshot
