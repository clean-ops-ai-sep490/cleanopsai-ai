import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ENVIRONMENT_WEIGHTS = {
    "OUTDOOR": 1.0,
    "GLASS": 1.2,
    "BASEMENT": 1.1,
    "LOBBY_CORRIDOR_ELEVATOR": 1.2,
    "RESTROOM": 1.3,
    "HOSPITAL_OR": 1.8,
}

CONTAMINATION_TO_ID = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}


@dataclass
class Sample:
    before_image: Path
    after_image: Path
    mask: Path
    contamination_id: int
    environment: str
    gps_distance_m: float
    timestamp_delta_sec: float
    device_match: bool
    ssim_anchor: float
    worker_override: bool


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def context_precheck_score(
    gps_distance_m: float,
    timestamp_delta_sec: float,
    device_match: bool,
    ssim_anchor: float,
    worker_override: bool,
) -> float:
    """Fail-fast precheck score in [0, 1], based on context consistency."""
    gps_score = 1.0 if gps_distance_m <= 8.0 else max(0.0, 1.0 - (gps_distance_m - 8.0) / 40.0)
    ts_score = 1.0 if timestamp_delta_sec <= 1800.0 else max(0.0, 1.0 - (timestamp_delta_sec - 1800.0) / 7200.0)
    device_score = 1.0 if device_match else 0.7
    ssim_score = max(0.0, min(1.0, ssim_anchor))

    combined = 0.30 * gps_score + 0.20 * ts_score + 0.20 * device_score + 0.30 * ssim_score
    if worker_override:
        combined *= 0.8
    return float(max(0.0, min(1.0, combined)))


class QualityScoringDataset(Dataset):
    """
    Expected metadata JSONL schema (one row per before/after pair):
    {
      "before_image": "images/train/before_x.jpg",
      "after_image": "images/train/after_x.jpg",
      "mask": "masks/train/after_x.png",
      "contamination_level": "LOW|MEDIUM|HIGH",
      "environment": "OUTDOOR|GLASS|BASEMENT|LOBBY_CORRIDOR_ELEVATOR|RESTROOM|HOSPITAL_OR",
      "gps_distance_m": 3.4,
      "timestamp_delta_sec": 240,
      "device_match": true,
      "ssim_anchor": 0.92,
      "worker_override": false
    }

    Mask encoding:
      0 = background
      1 = detected_stains
      2 = dirt_coverage
      3 = abnormal_objects
    """

    def __init__(self, data_root: Path, jsonl_file: Path) -> None:
        self.data_root = data_root
        rows = _read_jsonl(jsonl_file)
        self.samples: List[Sample] = []

        for row in rows:
            contamination_level = str(row["contamination_level"]).upper()
            contamination_id = CONTAMINATION_TO_ID[contamination_level]

            sample = Sample(
                before_image=data_root / str(row["before_image"]),
                after_image=data_root / str(row["after_image"]),
                mask=data_root / str(row["mask"]),
                contamination_id=contamination_id,
                environment=str(row["environment"]).upper(),
                gps_distance_m=float(row.get("gps_distance_m", 0.0)),
                timestamp_delta_sec=float(row.get("timestamp_delta_sec", 0.0)),
                device_match=bool(row.get("device_match", True)),
                ssim_anchor=float(row.get("ssim_anchor", 1.0)),
                worker_override=bool(row.get("worker_override", False)),
            )
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        before = np.asarray(Image.open(sample.before_image).convert("RGB"), dtype=np.float32) / 255.0
        after = np.asarray(Image.open(sample.after_image).convert("RGB"), dtype=np.float32) / 255.0

        x = np.concatenate([before, after], axis=2)
        x_t = torch.from_numpy(x).permute(2, 0, 1)

        mask = np.asarray(Image.open(sample.mask).convert("L"), dtype=np.int64)
        stain = (mask == 1).astype(np.float32)
        dirt = (mask == 2).astype(np.float32)
        abnormal = (mask == 3).astype(np.float32)
        seg_t = torch.from_numpy(np.stack([stain, dirt, abnormal], axis=0))

        precheck = context_precheck_score(
            gps_distance_m=sample.gps_distance_m,
            timestamp_delta_sec=sample.timestamp_delta_sec,
            device_match=sample.device_match,
            ssim_anchor=sample.ssim_anchor,
            worker_override=sample.worker_override,
        )

        env_weight = float(ENVIRONMENT_WEIGHTS.get(sample.environment, 1.0))
        sample_weight = max(0.2, precheck) * env_weight

        return {
            "x": x_t,
            "seg": seg_t,
            "contamination": torch.tensor(sample.contamination_id, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final AI quality scoring model")
    parser.add_argument("--data-root", default="data/scoring_final", help="Root folder for images/masks/metadata")
    parser.add_argument("--train-jsonl", default="train.jsonl")
    parser.add_argument("--val-jsonl", default="val.jsonl")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="models/quality_scoring_resnet50_unet.pt")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_loader(dataset: QualityScoringDataset, batch: int, workers: int, shuffle: bool) -> DataLoader:
    if not shuffle:
        return DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)

    weights = []
    for i in range(len(dataset)):
        item = dataset[i]
        weights.append(float(item["sample_weight"].item()))

    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=workers)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    seg_loss_fn: nn.Module,
    cls_loss_fn: nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    model.eval()
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            seg_target = batch["seg"].to(device)
            cls_target = batch["contamination"].to(device)
            w = batch["sample_weight"].to(device)

            seg_logits, cls_logits = model(x)
            seg_loss = seg_loss_fn(seg_logits, seg_target)
            cls_loss = cls_loss_fn(cls_logits, cls_target)
            loss = (seg_loss + cls_loss) * w.mean()

            total_seg_loss += float(seg_loss.item())
            total_cls_loss += float(cls_loss.item())
            total_batches += 1

    if total_batches == 0:
        return float("inf"), float("inf"), float("inf")

    avg_seg = total_seg_loss / total_batches
    avg_cls = total_cls_loss / total_batches
    return avg_seg + avg_cls, avg_seg, avg_cls


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data_root)
    train_ds = QualityScoringDataset(data_root=data_root, jsonl_file=data_root / args.train_jsonl)
    val_ds = QualityScoringDataset(data_root=data_root, jsonl_file=data_root / args.val_jsonl)

    train_loader = build_loader(train_ds, args.batch, args.num_workers, shuffle=True)
    val_loader = build_loader(val_ds, args.batch, args.num_workers, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=6,
        classes=3,
        activation=None,
        aux_params={
            "pooling": "avg",
            "dropout": 0.2,
            "classes": 3,
        },
    ).to(device)

    seg_loss_fn = nn.BCEWithLogitsLoss()
    cls_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            seg_target = batch["seg"].to(device)
            cls_target = batch["contamination"].to(device)
            w = batch["sample_weight"].to(device)

            optimizer.zero_grad()
            seg_logits, cls_logits = model(x)

            seg_loss = seg_loss_fn(seg_logits, seg_target)
            cls_loss = cls_loss_fn(cls_logits, cls_target)
            loss = (seg_loss + cls_loss) * w.mean()

            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item())
            train_batches += 1

        avg_train = total_train_loss / max(1, train_batches)
        avg_val, avg_val_seg, avg_val_cls = evaluate(model, val_loader, seg_loss_fn, cls_loss_fn, device)

        print(
            f"Epoch {epoch:03d} | train={avg_train:.4f} | val={avg_val:.4f} "
            f"(seg={avg_val_seg:.4f}, cls={avg_val_cls:.4f})"
        )

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), output_path)
            print(f"Saved better checkpoint to {output_path}")


if __name__ == "__main__":
    main()
