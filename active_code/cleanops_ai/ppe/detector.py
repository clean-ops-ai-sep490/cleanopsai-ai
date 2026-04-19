from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from ultralytics import YOLO

from active_code.cleanops_ai.config import PPE_MODEL_PATH


DetectionPayload = dict[str, Any]


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    image_index: int

    def as_payload(self) -> DetectionPayload:
        return {
            "name": self.name,
            "confidence": round(self.confidence, 1),
            "image_index": self.image_index,
        }


def normalize_confidence_threshold(min_confidence: float) -> float:
    return min_confidence * 100 if min_confidence <= 1 else min_confidence


@lru_cache(maxsize=1)
def load_model(model_path: str | Path = PPE_MODEL_PATH) -> YOLO:
    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved_model_path}")

    return YOLO(str(resolved_model_path))


def _load_image_from_url(image_url: str) -> Image.Image:
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def detect_from_image_url(
    image_url: str,
    min_confidence: float,
    image_index: int = 0,
) -> tuple[dict[str, float], list[DetectionPayload]]:
    confidence_threshold = normalize_confidence_threshold(min_confidence)
    model = load_model()
    image = _load_image_from_url(image_url)

    best_by_name: dict[str, Detection] = {}

    results = model(image)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf) * 100
            class_name = str(model.names[class_id]).lower()

            if confidence < confidence_threshold:
                continue

            current_best = best_by_name.get(class_name)
            if current_best is None or confidence > current_best.confidence:
                best_by_name[class_name] = Detection(
                    name=class_name,
                    confidence=confidence,
                    image_index=image_index,
                )

    detected_dict = {
        detection.name: detection.confidence
        for detection in best_by_name.values()
    }
    detected_list = [
        detection.as_payload()
        for detection in sorted(best_by_name.values(), key=lambda item: item.name)
    ]
    return detected_dict, detected_list
