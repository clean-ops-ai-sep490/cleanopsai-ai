# cleanopsai-ai

AI service for **CleanOpsAI** — exposes a trained ML model via a FastAPI REST API.

## Project Structure

```
cleanopsai-ai/
├── app/
│   ├── api/v1/          # Route handlers (health, prediction)
│   ├── core/            # App configuration (pydantic-settings)
│   ├── models/trained/  # Saved model artefacts (model.joblib)
│   ├── schemas/         # Pydantic request/response models
│   ├── services/        # Model loading & inference logic
│   └── main.py          # FastAPI application factory
├── tests/               # pytest test suite
├── train.py             # Example training script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This saves `app/models/trained/model.joblib`.  Replace the demo data in
`train.py` with your own dataset and feature engineering.

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive API documentation.

### 4. Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check & model status |
| POST | `/api/v1/prediction/predict` | Run model inference |

### Example prediction request

```bash
curl -X POST http://localhost:8000/api/v1/prediction/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, -0.5, 0.3, 0.8, -1.1]}'
```

## Running Tests

```bash
pytest tests/
```
