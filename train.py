"""
train.py — Example training script for CleanOpsAI.

Run:
    python train.py

This script trains a simple scikit-learn classifier on demo data and saves
the model to app/models/trained/model.joblib so that the API can load it.
Replace the data loading and model configuration with your own.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_OUTPUT = Path("app/models/trained/model.joblib")

# ---------------------------------------------------------------------------
# Replace the section below with your own data loading and feature engineering
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
X = rng.standard_normal((200, 5))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_OUTPUT)
print(f"Model saved to {MODEL_OUTPUT}")
