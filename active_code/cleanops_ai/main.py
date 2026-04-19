from __future__ import annotations

from fastapi import FastAPI, Query

from active_code.cleanops_ai.ppe import detect_from_image_url, evaluate_ppe_payload
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

    return app


app = create_app()
