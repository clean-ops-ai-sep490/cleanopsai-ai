from fastapi import FastAPI, Query

from ai_modules.ppe.service import detect_from_image_url, evaluate_ppe_payload
from ai_modules.scoring.service import SCORER
from ai_modules.schemas import AIRequest, QuickTestRequest, SanitationScoreRequest, SanitationScoreResponse

app = FastAPI(title="CleanOps AI Service")

@app.post("/api/ai/evaluate_ppe")
async def evaluate_ppe(req: AIRequest):
    return evaluate_ppe_payload(
        image_urls=req.image_urls,
        required_objects=req.required_objects,
        min_confidence=req.min_confidence,
    )

@app.get("/api/ai/test_detect")
async def test_detect(
    image_url: str = Query(..., description="Image URL to test quickly"),
    min_confidence: float = Query(0.25, ge=0, le=100, description="Confidence threshold (0-1 or 0-100)"),
):
    """Quick test endpoint: paste one image URL and see what objects are detected."""
    try:
        _, detected_list = detect_from_image_url(
            image_url=image_url,
            min_confidence=min_confidence,
            image_index=0,
        )
        return {
            "image_url": image_url,
            "detected_count": len(detected_list),
            "detected_items": detected_list,
            "detected_names": sorted(list({item["name"] for item in detected_list}))
        }
    except Exception as e:
        return {
            "image_url": image_url,
            "detected_count": 0,
            "detected_items": [],
            "detected_names": [],
            "error": str(e)
        }

@app.post("/api/ai/test_detect")
async def test_detect_post(req: QuickTestRequest):
    """Quick test endpoint for tools that prefer JSON body."""
    return await test_detect(image_url=req.image_url, min_confidence=req.min_confidence)


@app.post("/api/ai/quality-score", response_model=SanitationScoreResponse)
async def evaluate_sanitation(req: SanitationScoreRequest):
    """Evaluate before/after sanitation quality based on segmentation scoring model."""
    return SCORER.evaluate(before_url=req.before_img, after_url=req.after_img)