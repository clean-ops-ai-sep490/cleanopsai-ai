import os
from io import BytesIO
from typing import Dict, Tuple

import numpy as np
import requests
from PIL import Image, ImageFilter, ImageStat

try:
    import segmentation_models_pytorch as smp
    import torch
except Exception:  # noqa: BLE001
    smp = None
    torch = None


class SanitationScorer:
    """Compute before/after sanitation metrics from segmentation outputs."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.model = None
        if smp is not None and torch is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=3,
                classes=2,
            ).to(self.device)
            checkpoint_path = os.getenv("SANITATION_MODEL_PATH", "models/sanitation_unet_best.pt")
            if os.path.exists(checkpoint_path):
                state = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.eval()

    @staticmethod
    def _load_image(url: str) -> Image.Image:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    @staticmethod
    def _confidence_from_image_pair(before: Image.Image, after: Image.Image) -> float:
        before_blur = ImageStat.Stat(before.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]
        after_blur = ImageStat.Stat(after.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]

        before_luma = ImageStat.Stat(before.convert("L")).mean[0]
        after_luma = ImageStat.Stat(after.convert("L")).mean[0]

        sharpness_score = min(100.0, ((before_blur + after_blur) / 2.0) / 20.0)
        brightness_diff = abs(before_luma - after_luma)
        alignment_penalty = min(20.0, brightness_diff / 4.0)

        return round(max(40.0, min(99.0, sharpness_score + 20.0 - alignment_penalty)), 1)

    def _predict_masks(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(image, dtype=np.float32) / 255.0

        if self.model is None or torch is None:
            # Fallback heuristic when model is not available.
            gray = np.mean(arr, axis=2)
            stain_mask = gray < 0.35
            dirt_mask = np.logical_and(gray >= 0.35, gray < 0.6)
            return stain_mask.astype(np.uint8), dirt_mask.astype(np.uint8)

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        stain_mask = (probs[0] > 0.5).astype(np.uint8)
        dirt_mask = (probs[1] > 0.5).astype(np.uint8)
        return stain_mask, dirt_mask

    @staticmethod
    def _stain_count(stain_mask: np.ndarray) -> int:
        stain_pixels = int(stain_mask.sum())
        # Coarse blob proxy for PoC: tune divisor after collecting real data.
        return max(0, int(round(stain_pixels / 1800.0)))

    @staticmethod
    def _coverage_percent(dirt_mask: np.ndarray) -> float:
        total = dirt_mask.shape[0] * dirt_mask.shape[1]
        if total == 0:
            return 0.0
        return round(float(dirt_mask.sum()) / float(total) * 100.0, 1)

    def evaluate(self, before_url: str, after_url: str) -> Dict[str, object]:
        before_img = self._load_image(before_url)
        after_img = self._load_image(after_url)

        before_stain_mask, before_dirt_mask = self._predict_masks(before_img)
        after_stain_mask, after_dirt_mask = self._predict_masks(after_img)

        before_stains = self._stain_count(before_stain_mask)
        after_stains = self._stain_count(after_stain_mask)
        before_coverage = self._coverage_percent(before_dirt_mask)
        after_coverage = self._coverage_percent(after_dirt_mask)

        confidence = self._confidence_from_image_pair(before_img, after_img)

        return {
            "confidence_score": confidence,
            "before": {"stains": before_stains, "coverage": before_coverage},
            "after": {"stains": after_stains, "coverage": after_coverage},
        }


SCORER = SanitationScorer()
