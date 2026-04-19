import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from active_code.cleanops_ai.config import (
    PPE_BASE_MODEL_PATH,
    PPE_DATA_MANIFEST_PATH,
    TRAINING_REPORTS_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPE detection model (YOLO)")
    parser.add_argument("--data", default=str(PPE_DATA_MANIFEST_PATH), help="YOLO dataset yaml")
    parser.add_argument("--base-model", default=str(PPE_BASE_MODEL_PATH), help="Base checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default=str(TRAINING_REPORTS_DIR))
    parser.add_argument("--name", default="ppe_master_model")
    parser.add_argument(
        "--output",
        default=str(PPE_BASE_MODEL_PATH.parent / "ppe_detector.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_model_path = Path(args.base_model)
    data_manifest_path = Path(args.data)
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model checkpoint not found: {base_model_path}")
    if not data_manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {data_manifest_path}")

    model = YOLO(str(base_model_path))
    result = model.train(
        data=str(data_manifest_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(Path(args.project)),
        name=args.name,
    )

    best_checkpoint = Path(result.save_dir) / "weights" / "best.pt"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, output_path)

    print(f"Best checkpoint copied to: {output_path}")


if __name__ == "__main__":
    main()
