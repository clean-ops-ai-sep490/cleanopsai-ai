from typing import List

from pydantic import BaseModel


class AIRequest(BaseModel):
    image_urls: List[str]
    validation_type: str = "all_required"
    required_objects: List[str]
    min_confidence: float = 0.25


class QuickTestRequest(BaseModel):
    image_url: str
    min_confidence: float = 0.25
