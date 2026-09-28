"""Train a compact five-class EfficientNet-B0 baseline on APTOS 2019.

Example:
    python train.py --epochs 5 --image-size 384
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent


class AptosDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(self.image_dir / f'{row["id_code"]}.png').convert("RGB")
        return self.transform(image), int(row["diagnosis"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=0 if torch.cuda.is_available() is False else 2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    seed_everything(26038)
    csv_path = ROOT / "data" / "raw" / "aptos" / "train.csv"
    image_dir = ROOT / "data" / "raw" / "aptos" / "train_images"
    if not csv_path.exists() or not image_dir.exists():
        raise SystemExit("APTOS data not found under data/raw/aptos.")

    frame = pd.read_csv(csv_path)
    train_frame, validation_frame = train_test_split(
        frame, test_size=0.2, random_state=26038, stratify=frame["diagnosis"]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomRotation(12),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_data = AptosDataset(train_frame, image_dir, train_transform)
    validation_data = AptosDataset(validation_frame, image_dir, validation_transform)
    class_counts = train_frame["diagnosis"].value_counts().sort_index().to_numpy()
    sample_weights = 1.0 / class_counts[train_frame["diagnosis"].to_numpy()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = None if args.no_pretrained else models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_kappa = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        labels_all, predictions_all = evaluate(model, validation_loader, device)
        kappa = cohen_kappa_score(labels_all, predictions_all, weights="quadratic")
        print(f"epoch={epoch} loss={running_loss / len(train_data):.4f} val_qwk={kappa:.4f}")
        print(confusion_matrix(labels_all, predictions_all))
        print(classification_report(labels_all, predictions_all, digits=3, zero_division=0))
        if kappa > best_kappa:
            best_kappa = kappa
            output = ROOT / "models" / "dr_model.pth"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_size": args.image_size,
                    "model_version": "efficientnet-b0-aptos-hackathon-v1",
                    "validation_qwk": float(kappa),
                },
                output,
            )
            print(f"saved={output}")


def evaluate(model, loader, device):
    model.eval()
    labels_all, predictions_all = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions.tolist())
    return labels_all, predictions_all


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()

