# cleanopsai-ai

AI service for CleanOps PoC focused on PPE detection using YOLO.

## Project Structure

```text
.
|-- ai_modules/
|   |-- schemas.py                  # API request models
|   `-- ppe/
|       |-- service.py              # PPE inference logic
|       |-- train.py                # PPE training entrypoint
|       |-- notebooks/              # PPE notebooks
|       |-- test_api_skeleton.py    # PPE API smoke test script
|       `-- tools/                  # PPE data helper scripts
|-- main.py                         # FastAPI entrypoint
`-- requirements.txt
```

## API Endpoints

- `POST /api/ai/evaluate_ppe`
- `GET /api/ai/test_detect`
- `POST /api/ai/test_detect`

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Train PPE Model (YOLO)

```bash
python ai_modules/ppe/train.py \
	--data master_dataset.yaml \
	--base-model best_ppe_model_v1.pt \
	--epochs 50 \
	--output best_ppe_model_v2_incremental.pt
```
