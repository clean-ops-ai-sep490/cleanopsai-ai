# cleanopsai-ai

CleanOps AI service for PPE detection with a reduced repository layout centered on one active model, one canonical dataset manifest, and one useful smoke test.

## Repository Layout

```text
.
|-- active_code/                    # Code that backs the running API
|   `-- cleanops_ai/
|       |-- config.py               # Centralized paths and model settings
|       |-- main.py                 # FastAPI app factory and routes
|       |-- schemas.py              # API request models
|       `-- ppe/
|           |-- detector.py         # Model loading + image inference
|           |-- service.py          # PPE evaluation workflow
|           `-- train.py            # Training entrypoint
|-- artifacts/
|   |-- manifests/                 # Dataset manifests used for training
|   `-- models/                    # Canonical local checkpoints
|-- experiments/
|   `-- ppe/                       # Lightweight smoke test for the API
|-- datasets/
|   `-- ppe_dataset/               # Canonical local training dataset
|-- main.py                        # Stable compatibility entrypoint
`-- requirements.txt
```

## Active Model

The API serves `artifacts/models/ppe_detector.pt` by default.

You can override it with `PPE_MODEL_PATH` if you want to test a different checkpoint without editing the code.

The canonical training manifest is `artifacts/manifests/ppe_dataset.yaml`, which points at `datasets/ppe_dataset`.

## API Endpoints

- `GET /health`
- `POST /api/ai/evaluate_ppe`
- `GET /api/ai/test_detect`
- `POST /api/ai/test_detect`

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Train the PPE Model

```bash
python -m active_code.cleanops_ai.ppe.train \
  --data artifacts/manifests/ppe_dataset.yaml \
  --base-model artifacts/models/ppe_detector.pt \
  --epochs 50 \
  --output artifacts/models/ppe_detector.pt
```

## Smoke Test the API

```bash
python experiments/ppe/ppe_api_smoke_test.py
```
