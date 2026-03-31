from io import BytesIO
from typing import Dict, List, Tuple

import requests
from PIL import Image
from ultralytics import YOLO

# Load latest PPE checkpoint once at startup.
PPE_MODEL = YOLO("best_ppe_model_v2_incremental.pt")


def detect_from_image_url(
    image_url: str,
    min_confidence: float,
    image_index: int = 0,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Load an image URL and return detections filtered by confidence."""
    detected_dict: Dict[str, float] = {}
    detected_list: List[Dict[str, float]] = []

    confidence_threshold = min_confidence * 100 if min_confidence <= 1 else min_confidence

    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")

    results = PPE_MODEL(img)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf) * 100
            class_name = str(PPE_MODEL.names[class_id]).lower()

            if confidence >= confidence_threshold:
                if class_name not in detected_dict or confidence > detected_dict[class_name]:
                    detected_dict[class_name] = confidence
                    detected_list.append(
                        {
                            "name": class_name,
                            "confidence": round(confidence, 1),
                            "image_index": image_index,
                        }
                    )

    return detected_dict, detected_list


def evaluate_ppe_payload(image_urls: List[str], required_objects: List[str], min_confidence: float):
    """Evaluate PPE requirements across one or many images."""
    detected_dict: Dict[str, float] = {}
    detected_list: List[Dict[str, float]] = []

    normalized_required_objects = [obj.strip().lower() for obj in required_objects]

    for idx, url in enumerate(image_urls):
        try:
            per_image_dict, per_image_list = detect_from_image_url(
                image_url=url,
                min_confidence=min_confidence,
                image_index=idx,
            )
            for key, val in per_image_dict.items():
                if key not in detected_dict or val > detected_dict[key]:
                    detected_dict[key] = val
            detected_list.extend(per_image_list)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load image {url}: {exc}")

    missing_items = [obj for obj in normalized_required_objects if obj not in detected_dict]

    status = "PASS" if len(missing_items) == 0 else "FAIL"
    message = "Meets requirements." if status == "PASS" else f"Missing items: {', '.join(missing_items)}"

    return {
        "status": status,
        "message": message,
        "detected_items": detected_list,
        "missing_items": missing_items,
    }
