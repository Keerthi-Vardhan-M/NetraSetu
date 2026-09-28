"""DR model loading and transparent dataset-label demonstration fallback."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from src.grades import GRADE_LABELS


class DRGrader:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.checkpoint_path = project_root / "models" / "dr_model.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.model_version = "no-trained-checkpoint"
        self.input_size = 384
        self.demo_labels = self._load_demo_labels()
        self._load_checkpoint()
        self.transform = self._build_transform()

    def _build_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.input_size = int(checkpoint.get("input_size", 384))
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device).eval()
        self.model = model
        self.model_version = checkpoint.get("model_version", "efficientnet-b0-hackathon")

    def predict(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        if self.model is not None:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            grade = int(probabilities.argmax())
            confidence = float(probabilities[grade])
            return self._result(grade, probabilities, confidence, "trained-model")

        image_id = image_path.stem
        if image_id in self.demo_labels:
            grade = self.demo_labels[image_id]
            probabilities = np.full(5, 0.025, dtype=float)
            probabilities[grade] = 0.90
            return self._result(grade, probabilities, 0.90, "aptos-label-demo")

        return {
            "grade": None,
            "label": "Manual review required",
            "probabilities": [],
            "confidence": 0.0,
            "low_confidence": True,
            "mode": "no-model",
            "model_version": self.model_version,
            "warning": "No trained checkpoint is installed; this uploaded image was not graded.",
        }

    def _result(self, grade: int, probabilities: np.ndarray, confidence: float, mode: str) -> dict:
        return {
            "grade": grade,
            "label": GRADE_LABELS[grade],
            "probabilities": [round(float(value), 4) for value in probabilities],
            "confidence": round(confidence, 4),
            "low_confidence": confidence < 0.55,
            "mode": mode,
            "model_version": self.model_version if mode == "trained-model" else "aptos-ground-truth-demo",
            "warning": (
                "Demonstration result uses the APTOS reference label, not a trained model prediction."
                if mode == "aptos-label-demo"
                else ""
            ),
        }

    def _load_demo_labels(self) -> dict[str, int]:
        csv_path = self.project_root / "data" / "raw" / "aptos" / "train.csv"
        if not csv_path.exists():
            return {}
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            return {row["id_code"]: int(row["diagnosis"]) for row in csv.DictReader(handle)}


@lru_cache(maxsize=1)
def get_grader(project_root: str) -> DRGrader:
    return DRGrader(Path(project_root))
