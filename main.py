from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load checkpoint fine-tuned moi nhat de test sau train.
model = YOLO("best_ppe_model_v2_incremental.pt") 

class AIRequest(BaseModel):
    image_urls: List[str]
    validation_type: str
    required_objects: List[str]
    min_confidence: float

class QuickTestRequest(BaseModel):
    image_url: str
    min_confidence: float = 0.25

def _detect_from_image_url(image_url: str, min_confidence: float, image_index: int = 0):
    """Load one image URL and return detections filtered by confidence."""
    detected_dict = {}
    detected_list = []

    confidence_threshold = min_confidence * 100 if min_confidence <= 1 else min_confidence

    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))

    results = model(img)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf) * 100
            class_name = str(model.names[class_id]).lower()

            if confidence >= confidence_threshold:
                if class_name not in detected_dict or confidence > detected_dict[class_name]:
                    detected_dict[class_name] = confidence
                    detected_list.append({
                        "name": class_name,
                        "confidence": round(confidence, 1),
                        "image_index": image_index
                    })

    return detected_dict, detected_list

@app.post("/api/ai/evaluate_ppe")
async def evaluate_ppe(req: AIRequest):
    detected_dict = {}
    detected_list = []

    # Accept both 0-1 and 0-100 styles for min_confidence.
    confidence_threshold = req.min_confidence * 100 if req.min_confidence <= 1 else req.min_confidence
    normalized_required_objects = [obj.strip().lower() for obj in req.required_objects]

    # 1. Tải và quét từng ảnh từ MinIO
    for idx, url in enumerate(req.image_urls):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            
            # Quét ảnh bằng YOLO
            results = model(img)

            # model(img) returns a list of Results; we process each result's boxes.
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf) * 100
                    class_name = str(model.names[class_id]).lower() # Ví dụ: 'person', 'tie', v.v.

                    if confidence >= confidence_threshold:
                        if class_name not in detected_dict or confidence > detected_dict[class_name]:
                            detected_dict[class_name] = confidence
                            detected_list.append({
                                "name": class_name,
                                "confidence": round(confidence, 1),
                                "image_index": idx
                            })
        except Exception as e:
            print(f"Failed to load image {url}: {e}")

    # 2. Logic kiểm tra động (So khớp những gì tìm thấy với yêu cầu của Manager)
    missing_items = []
    for req_obj in normalized_required_objects:
        if req_obj not in detected_dict:
            missing_items.append(req_obj)
            
    status = "PASS" if len(missing_items) == 0 else "FAIL"
    message = "Meets requirements." if status == "PASS" else f"Missing items: {', '.join(missing_items)}"

    return {
        "status": status,
        "message": message,
        "detected_items": detected_list,
        "missing_items": missing_items
    }

@app.get("/api/ai/test_detect")
async def test_detect(
    image_url: str = Query(..., description="Image URL to test quickly"),
    min_confidence: float = Query(0.25, ge=0, le=100, description="Confidence threshold (0-1 or 0-100)"),
):
    """Quick test endpoint: paste one image URL and see what objects are detected."""
    try:
        _, detected_list = _detect_from_image_url(
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