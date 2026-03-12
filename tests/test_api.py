"""Tests for the CleanOpsAI API."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

app = create_app()
client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "CleanOpsAI"
    assert "version" in data
    assert "model_loaded" in data


def test_predict_without_model_returns_503() -> None:
    """Without a trained model the endpoint should return 503."""
    response = client.post("/api/v1/prediction/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 503


def test_predict_with_model(tmp_path, monkeypatch) -> None:
    """With a loaded model the endpoint should return a prediction."""
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    from app.services import model_service as ms_module

    # Train and save a tiny model
    X = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]])
    y = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(X, y)
    model_file = tmp_path / "model.joblib"
    joblib.dump(clf, model_file)

    # Point the service at the temp model and reload
    monkeypatch.setattr(ms_module.model_service, "model_path", tmp_path)
    ms_module.model_service._model = None
    ms_module.model_service.load()

    response = client.post("/api/v1/prediction/predict", json={"features": [1.0, 0.0, 1.0]})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data


def test_docs_available() -> None:
    response = client.get("/docs")
    assert response.status_code == 200
