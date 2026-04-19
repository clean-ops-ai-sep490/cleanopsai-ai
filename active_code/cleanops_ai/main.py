from __future__ import annotations

from fastapi import FastAPI, Query, Response

from active_code.cleanops_ai.ppe import (
    detect_from_image_url,
    evaluate_ppe_payload,
    visualize_image_bytes_from_image_url,
    visualize_from_image_url,
)
from active_code.cleanops_ai.schemas import AIRequest, QuickTestRequest


def create_app() -> FastAPI:
    app = FastAPI(title="CleanOps AI Service")

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/ai/evaluate_ppe")
    async def evaluate_ppe(request: AIRequest) -> dict[str, object]:
        return evaluate_ppe_payload(
            image_urls=request.image_urls,
            required_objects=request.required_objects,
            min_confidence=request.min_confidence,
        )

    @app.get("/api/ai/test_detect")
    async def test_detect(
        image_url: str = Query(..., description="Image URL to test quickly"),
        min_confidence: float = Query(
            0.25,
            ge=0,
            le=100,
            description="Confidence threshold (0-1 or 0-100)",
        ),
    ) -> dict[str, object]:
        """Quick test endpoint: paste one image URL and see what objects are detected."""
        try:
            _, detected_items = detect_from_image_url(
                image_url=image_url,
                min_confidence=min_confidence,
                image_index=0,
            )
            return {
                "image_url": image_url,
                "detected_count": len(detected_items),
                "detected_items": detected_items,
                "detected_names": sorted({item["name"] for item in detected_items}),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "image_url": image_url,
                "detected_count": 0,
                "detected_items": [],
                "detected_names": [],
                "error": str(exc),
            }

    @app.post("/api/ai/test_detect")
    async def test_detect_post(request: QuickTestRequest) -> dict[str, object]:
        """Quick test endpoint for tools that prefer JSON body."""
        return await test_detect(
            image_url=request.image_url,
            min_confidence=request.min_confidence,
        )

    @app.get("/api/ai/test_detect_visualize")
    async def test_detect_visualize(
        image_url: str = Query(..., description="Image URL to test quickly"),
        min_confidence: float = Query(
            0.25,
            ge=0,
            le=100,
            description="Confidence threshold (0-1 or 0-100)",
        ),
    ) -> dict[str, object]:
        """Quick visualization endpoint: returns detections and an annotated image."""
        try:
            return visualize_from_image_url(
                image_url=image_url,
                min_confidence=min_confidence,
                image_index=0,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "image_url": image_url,
                "detected_count": 0,
                "detected_items": [],
                "detected_names": [],
                "annotated_image_base64": None,
                "annotated_image_data_url": None,
                "image_format": "png",
                "error": str(exc),
            }

    @app.post("/api/ai/test_detect_visualize")
    async def test_detect_visualize_post(request: QuickTestRequest) -> dict[str, object]:
        """Quick visualization endpoint for tools that prefer JSON body."""
        return await test_detect_visualize(
            image_url=request.image_url,
            min_confidence=request.min_confidence,
        )

    @app.get("/api/ai/test_detect_visualize_image")
    async def test_detect_visualize_image(
        image_url: str = Query(..., description="Image URL to test quickly"),
        min_confidence: float = Query(
            0.25,
            ge=0,
            le=100,
            description="Confidence threshold (0-1 or 0-100)",
        ),
    ) -> Response:
        """Quick visualization image endpoint: returns the annotated image directly."""
        image_bytes = visualize_image_bytes_from_image_url(
            image_url=image_url,
            min_confidence=min_confidence,
            image_index=0,
        )
        return Response(content=image_bytes, media_type="image/png")

    return app


app = create_app()
