from pydantic import BaseModel, Field


class AIRequest(BaseModel):
    image_urls: list[str] = Field(..., min_length=1)
    validation_type: str = "all_required"
    required_objects: list[str] = Field(..., min_length=1)
    min_confidence: float = 0.25


class QuickTestRequest(BaseModel):
    image_url: str
    min_confidence: float = 0.25
