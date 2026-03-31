import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SanitationDataset(Dataset):
    """Expect mask pixel values: 0=background, 1=stain, 2=dirt."""

    def __init__(self, image_dir: Path, mask_dir: Path) -> None:
        self.image_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{image_path.stem}.png"

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.int64)

        image_t = torch.from_numpy(image).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask)
        return image_t, mask_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sanitation scoring segmentation model")
    parser.add_argument("--data-root", default="data/sanitation", help="Root folder with images/ and masks/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="models/sanitation_unet_best.pt")
    return parser.parse_args()


def build_loader(data_root: Path, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    image_dir = data_root / "images" / split
    mask_dir = data_root / "masks" / split
    ds = SanitationDataset(image_dir=image_dir, mask_dir=mask_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += float(loss.item())
            total_batches += 1

    if total_batches == 0:
        return float("inf")
    return total_loss / total_batches


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data_root)
    train_loader = build_loader(data_root, "train", args.batch, True)
    val_loader = build_loader(data_root, "val", args.batch, False)

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_batches += 1

        avg_train = train_loss / max(1, train_batches)
        avg_val = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), output_path)
            print(f"Saved better checkpoint to {output_path}")


if __name__ == "__main__":
    main()
