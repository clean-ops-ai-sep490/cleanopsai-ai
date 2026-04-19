from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw
from ultralytics import YOLO

from active_code.cleanops_ai.config import PPE_MODEL_PATH


DetectionPayload = dict[str, Any]
BBoxPayload = dict[str, float]


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


def _serialize_bbox(box: Any) -> BBoxPayload:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return {
        "x1": round(float(x1), 1),
        "y1": round(float(y1), 1),
        "x2": round(float(x2), 1),
        "y2": round(float(y2), 1),
    }


def _collect_filtered_detections(
    image: Image.Image,
    min_confidence: float,
    image_index: int,
) -> list[DetectionPayload]:
    confidence_threshold = normalize_confidence_threshold(min_confidence)
    model = load_model()
    detections: list[DetectionPayload] = []

    results = model(image)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf) * 100
            if confidence < confidence_threshold:
                continue

            detections.append(
                {
                    "name": str(model.names[class_id]).lower(),
                    "confidence": round(confidence, 1),
                    "image_index": image_index,
                    "bbox": _serialize_bbox(box),
                }
            )

    return detections


def _encode_image_to_data_url(image: Image.Image) -> tuple[str, str]:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded, f"data:image/png;base64,{encoded}"


def _encode_image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_detection_boxes(image: Image.Image, detections: list[DetectionPayload]) -> Image.Image:
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    for detection in detections:
        bbox = detection["bbox"]
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
        label = f"{detection['name']} {detection['confidence']}%"

        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)

        text_bbox = draw.textbbox((x1, y1), label)
        text_height = text_bbox[3] - text_bbox[1]
        text_bottom = y1 if y1 > text_height + 8 else y1 + text_height + 8
        text_top = text_bottom - text_height - 8
        text_right = text_bbox[2] + 8

        draw.rectangle((x1, text_top, text_right, text_bottom), fill="lime")
        draw.text((x1 + 4, text_top + 4), label, fill="black")

    return annotated_image


def detect_from_image_url(
    image_url: str,
    min_confidence: float,
    image_index: int = 0,
) -> tuple[dict[str, float], list[DetectionPayload]]:
    image = _load_image_from_url(image_url)
    detections = _collect_filtered_detections(
        image=image,
        min_confidence=min_confidence,
        image_index=image_index,
    )
    best_by_name: dict[str, Detection] = {}
    for detection in detections:
        class_name = str(detection["name"])
        confidence = float(detection["confidence"])
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


def visualize_from_image_url(
    image_url: str,
    min_confidence: float,
    image_index: int = 0,
) -> dict[str, Any]:
    image = _load_image_from_url(image_url)
    detections = _collect_filtered_detections(
        image=image,
        min_confidence=min_confidence,
        image_index=image_index,
    )
    annotated_image = _draw_detection_boxes(image=image, detections=detections)
    encoded_image, image_data_url = _encode_image_to_data_url(annotated_image)

    return {
        "image_url": image_url,
        "detected_count": len(detections),
        "detected_items": detections,
        "detected_names": sorted({str(item["name"]) for item in detections}),
        "annotated_image_base64": encoded_image,
        "annotated_image_data_url": image_data_url,
        "image_format": "png",
    }


def visualize_image_bytes_from_image_url(
    image_url: str,
    min_confidence: float,
    image_index: int = 0,
) -> bytes:
    image = _load_image_from_url(image_url)
    detections = _collect_filtered_detections(
        image=image,
        min_confidence=min_confidence,
        image_index=image_index,
    )
    annotated_image = _draw_detection_boxes(image=image, detections=detections)
    return _encode_image_to_png_bytes(annotated_image)
