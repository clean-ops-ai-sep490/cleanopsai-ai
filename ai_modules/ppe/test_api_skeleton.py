"""
Skeleton script to test 10 PPE API cases.

How to use:
1) Start your FastAPI app (default: http://127.0.0.1:8000)
2) Fill image URLs and expected notes in TEST_CASES below
3) Run: python test_ppe_api_skeleton.py
"""

import json
import time
from typing import Any, Dict, List

import requests

BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/api/ai/evaluate_ppe"
TIMEOUT_SECONDS = 120


def make_case(
    case_id: int,
    name: str,
    image_urls: List[str],
    validation_type: str,
    required_objects: List[str],
    min_confidence: float,
    expected_note: str,
) -> Dict[str, Any]:
    return {
        "id": case_id,
        "name": name,
        "payload": {
            "image_urls": image_urls,
            "validation_type": validation_type,
            "required_objects": required_objects,
            "min_confidence": min_confidence,
        },
        "expected_note": expected_note,
    }


TEST_CASES: List[Dict[str, Any]] = [
    make_case(
        1,
        "Single image - basic helmet",
        ["https://m.media-amazon.com/images/I/610NgSONfyL.jpg"],
        "all_required",
        ["helmet"],
        0.5,
        "Expected: PASS when helmet is clearly visible.",
    ),
    make_case(
        2,
        "Single image - multiple PPE",
        ["https://p.turbosquid.com/ts-thumb/x6/WapKYa/ADsh2hHi/safetygear_pack_01/jpg/1452305856/1920x1080/fit_q87/532c0a8c904576b79c22bdec555c0c2270398efb/safetygear_pack_01.jpg"],
        "all_required",
        ["helmet", "safety_vest", "boots"],
        0.5,
        "Expected: PASS when all required objects exist.",
    ),
    make_case(
        3,
        "Single image - missing one PPE",
        ["https://media.istockphoto.com/id/688052986/photo/white-safety-helmet-and-safety-shoes-this-is-personal-protective-equipment-for-construction.jpg?s=1024x1024&w=is&k=20&c=pizMgU6KGu_5ndmvTaEVCUe0A3wri1dfBGOUy03NTs4="],
        "all_required",
        ["helmet", "safety_vest"],
        0.5,
        "Expected: FAIL when one required object is not detected.",
    ),
    make_case(
        4,
        "Two images - combined coverage",
        ["https://blueeagle-safety.com/wp-content/uploads/2018/10/HC31BL.jpg", "https://meowprintsg.b-cdn.net/wp-content/uploads/2024/03/09231246/CRSV3100-Reflective-Safety-Vest-With-Pocket-25-26-MODEL-2.jpg?class=mpwmgallery"],
        "all_required",
        ["helmet", "safety_vest"],
        0.5,
        "Expected: PASS if object A in image 1 and object B in image 2.",
    ),
    make_case(
        5,
        "Threshold test - 0 to 1 style",
        ["https://res.cloudinary.com/rsc/image/upload/w_1024/F9185655-01"],
        "all_required",
        ["helmet"],
        0.7,
        "Expected: PASS/FAIL depending on whether confidence >= 70%.",
    ),
    # make_case(
    #     6,
    #     "Threshold test - 0 to 100 style",
    #     ["REPLACE_WITH_IMAGE_URL_6"],
    #     "all_required",
    #     ["helmet"],
    #     70,
    #     "Expected: Same behavior as case 5.",
    # ),
    # make_case(
    #     7,
    #     "Very high threshold",
    #     ["REPLACE_WITH_IMAGE_URL_7"],
    #     "all_required",
    #     ["helmet"],
    #     99,
    #     "Expected: Often FAIL due to strict confidence threshold.",
    # ),
    # make_case(
    #     8,
    #     "Class naming mismatch",
    #     ["REPLACE_WITH_IMAGE_URL_8"],
    #     "all_required",
    #     ["vest"],
    #     0.5,
    #     "Expected: Useful to verify label naming consistency.",
    # ),
    # make_case(
    #     9,
    #     "Normalization test (spaces/case)",
    #     ["REPLACE_WITH_IMAGE_URL_9"],
    #     "all_required",
    #     ["  GLOVES  "],
    #     0.5,
    #     "Expected: Should behave like 'gloves'.",
    # ),
    # make_case(
    #     10,
    #     "One bad URL + one good URL",
    #     ["https://invalid-host.invalid/not-found.jpg", "REPLACE_WITH_IMAGE_URL_10"],
    #     "all_required",
    #     ["helmet"],
    #     0.5,
    #     "Expected: API should still return response from valid images.",
    # ),
]


def run_case(base_url: str, case: Dict[str, Any]) -> Dict[str, Any]:
    url = base_url.rstrip("/") + ENDPOINT
    payload = case["payload"]

    started = time.time()
    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
        elapsed_ms = int((time.time() - started) * 1000)

        body: Any
        try:
            body = response.json()
        except Exception:
            body = {"raw_text": response.text}

        return {
            "id": case["id"],
            "name": case["name"],
            "http_status": response.status_code,
            "elapsed_ms": elapsed_ms,
            "expected_note": case["expected_note"],
            "payload": payload,
            "response": body,
        }
    except Exception as exc:
        elapsed_ms = int((time.time() - started) * 1000)
        return {
            "id": case["id"],
            "name": case["name"],
            "http_status": None,
            "elapsed_ms": elapsed_ms,
            "expected_note": case["expected_note"],
            "payload": payload,
            "response": {"error": str(exc)},
        }


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 88)
    print("PPE API TEST SUMMARY")
    print("=" * 88)

    for item in results:
        print(f"Case {item['id']:02d} | {item['name']}")
        print(f"- HTTP: {item['http_status']} | Time: {item['elapsed_ms']} ms")
        print(f"- Note: {item['expected_note']}")

        response = item["response"]
        if isinstance(response, dict):
            status = response.get("status")
            message = response.get("message")
            missing = response.get("missing_items")
            if status is not None:
                print(f"- API status: {status}")
            if message is not None:
                print(f"- API message: {message}")
            if missing is not None:
                print(f"- Missing: {missing}")
        print("-" * 88)


def main() -> None:
    print(f"Running {len(TEST_CASES)} cases against: {BASE_URL}{ENDPOINT}")
    results = [run_case(BASE_URL, case) for case in TEST_CASES]
    print_summary(results)

    print("\nFull JSON result:")
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
