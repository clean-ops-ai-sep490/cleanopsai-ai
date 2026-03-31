import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPE detection model (YOLO)")
    parser.add_argument("--data", default="master_dataset.yaml", help="YOLO dataset yaml")
    parser.add_argument("--base-model", default="best_ppe_model_v1.pt", help="Base checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs_master")
    parser.add_argument("--name", default="ppe_master_model")
    parser.add_argument("--output", default="best_ppe_model_v2_incremental.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.base_model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    best_checkpoint = Path(result.save_dir) / "weights" / "best.pt"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, output_path)

    print(f"Best checkpoint copied to: {output_path}")


if __name__ == "__main__":
    main()
