import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and running inference with a trained ML model."""

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self._model: Any = None

    def load(self) -> bool:
        """Load the model from disk. Returns True if successful."""
        model_file = self.model_path / "model.joblib"
        if not model_file.exists():
            logger.warning("Model file not found at %s", model_file)
            return False
        try:
            self._model = joblib.load(model_file)
            logger.info("Model loaded from %s", model_file)
            return True
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, features: list[float]) -> dict[str, Any]:
        """Run inference and return prediction with optional probability."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")
        X = np.array(features).reshape(1, -1)
        prediction = self._model.predict(X)[0]
        probability: float | None = None
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)[0]
            probability = float(max(proba))
        return {"prediction": prediction.item() if hasattr(prediction, "item") else prediction, "probability": probability}


model_service = ModelService(model_path=os.getenv("MODEL_PATH", "app/models/trained"))
