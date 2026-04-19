from __future__ import annotations

import logging
from typing import Any

from active_code.cleanops_ai.ppe.detector import detect_from_image_url

logger = logging.getLogger(__name__)


def _normalize_required_objects(required_objects: list[str]) -> list[str]:
    return [item.strip().lower() for item in required_objects if item.strip()]


def _merge_detected_items(
    aggregated_confidences: dict[str, float],
    detected_items: list[dict[str, Any]],
    per_image_confidences: dict[str, float],
    per_image_items: list[dict[str, Any]],
) -> None:
    for label, confidence in per_image_confidences.items():
        previous_confidence = aggregated_confidences.get(label)
        if previous_confidence is None or confidence > previous_confidence:
            aggregated_confidences[label] = confidence

    detected_items.extend(per_image_items)


def evaluate_ppe_payload(
    image_urls: list[str],
    required_objects: list[str],
    min_confidence: float,
) -> dict[str, Any]:
    """Evaluate PPE requirements across one or many images."""
    aggregated_confidences: dict[str, float] = {}
    detected_items: list[dict[str, Any]] = []
    failed_images: list[dict[str, Any]] = []
    normalized_required_objects = _normalize_required_objects(required_objects)

    for image_index, image_url in enumerate(image_urls):
        try:
            per_image_confidences, per_image_items = detect_from_image_url(
                image_url=image_url,
                min_confidence=min_confidence,
                image_index=image_index,
            )
            _merge_detected_items(
                aggregated_confidences=aggregated_confidences,
                detected_items=detected_items,
                per_image_confidences=per_image_confidences,
                per_image_items=per_image_items,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process image '%s': %s", image_url, exc)
            failed_images.append(
                {
                    "image_url": image_url,
                    "image_index": image_index,
                    "error": str(exc),
                }
            )

    missing_items = [
        required_item
        for required_item in normalized_required_objects
        if required_item not in aggregated_confidences
    ]
    status = "PASS" if not missing_items else "FAIL"
    message = (
        "Meets requirements."
        if status == "PASS"
        else f"Missing items: {', '.join(missing_items)}"
    )

    response: dict[str, Any] = {
        "status": status,
        "message": message,
        "detected_items": detected_items,
        "missing_items": missing_items,
    }
    if failed_images:
        response["failed_images"] = failed_images

    return response
