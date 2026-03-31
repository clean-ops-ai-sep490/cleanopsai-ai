# cleanopsai-ai

AI service for CleanOps PoC with 2 model pipelines in one repository:

- PPE detection model (YOLO)
- Sanitation scoring model (U-Net ResNet50)

## Project Structure

```text
.
|-- ai_modules/
|   |-- schemas.py                  # API request/response models
|   |-- ppe/
|   |   |-- service.py              # PPE inference logic
|   |   |-- train.py                # Train file #1 (PPE)
|   |   |-- notebooks/              # PPE training notebooks
|   |   |-- test_api_skeleton.py    # PPE API smoke test script
|   |   `-- tools/                  # PPE data helper scripts (Roboflow checks)
|   `-- scoring/
|       |-- service.py              # Sanitation scoring inference logic
|       |-- train_baseline.py       # Baseline scoring trainer
|       `-- train_quality_scoring.py # Final scoring trainer (requirements)
|-- main.py                         # FastAPI entrypoint
`-- requirements.txt
```

## API Endpoints

- `POST /api/ai/evaluate_ppe`
- `GET /api/ai/test_detect`
- `POST /api/ai/test_detect`
- `POST /api/ai/quality-score`

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

## Train Sanitation Scoring Model (Baseline U-Net)

Dataset expected layout:

```text
data/sanitation/
|-- images/
|   |-- train/
|   `-- val/
`-- masks/
		|-- train/
		`-- val/
```

Mask pixel values:

- `0` = background
- `1` = stain
- `2` = dirt

```bash
python ai_modules/scoring/train_baseline.py \
	--data-root data/sanitation \
	--epochs 30 \
	--output models/sanitation_unet_best.pt
```

## Train Final Quality Scoring Model (From Requirements)

This trainer supports:

- before/after pair input (6 channels)
- segmentation labels: detected_stains, dirt_coverage, abnormal_objects
- contamination_level classification: LOW / MEDIUM / HIGH
- fail-fast context weighting from GPS, timestamp, device match, and SSIM anchor
- environment-aware weighting (OUTDOOR, GLASS, BASEMENT, LOBBY/CORRIDOR/ELEVATOR, RESTROOM, HOSPITAL_OR)

Dataset schema file:

- ai_modules/scoring/configs/quality_scoring_dataset_schema.json

Train command:

```bash
python ai_modules/scoring/train_quality_scoring.py \
	--data-root data/scoring_final \
	--train-jsonl train.jsonl \
	--val-jsonl val.jsonl \
	--epochs 35 \
	--output models/quality_scoring_resnet50_unet.pt
```

## Use Sanitation Checkpoint in API

Set environment variable before starting API:

```bash
set SANITATION_MODEL_PATH=models/sanitation_unet_best.pt
```
