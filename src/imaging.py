"""Fundus image validation and lightweight quality assessment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    return image


def assess_quality(path: str | Path) -> dict:
    """Return explainable capture-quality checks.

    Thresholds are intentionally conservative demo defaults and must be calibrated
    against the camera used by a real screening programme.
    """

    image = read_image(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if min(height, width) < 224:
        return _quality(False, ["Image resolution is too low"], 0.0, 0.0, 0.0)

    retinal_mask = gray > 12
    coverage = float(retinal_mask.mean())
    pixels = gray[retinal_mask] if retinal_mask.any() else gray.reshape(-1)
    brightness = float(pixels.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare = float((pixels > 245).mean())

    problems: list[str] = []
    # APTOS images are strongly compressed and naturally smooth, so the raw
    # Laplacian scale is much lower than ordinary phone photographs. This
    # conservative threshold rejects the bottom tail and should be calibrated
    # again for the actual fundus camera used at the event.
    if blur_score < 5:
        problems.append("Image appears blurred or shaky")
    if brightness < 35:
        problems.append("Image is too dark")
    elif brightness > 220:
        problems.append("Image is overexposed")
    if coverage < 0.28:
        problems.append("Retinal field occupies too little of the image")
    if glare > 0.08:
        problems.append("Strong glare or reflection detected")

    return _quality(not problems, problems, blur_score, brightness, coverage, glare)


def image_fingerprint(path: str | Path) -> str:
    image = read_image(path)
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (16, 16))
    return hashlib.sha256(small.tobytes()).hexdigest()


def crop_fundus(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = cv2.findNonZero((gray > 10).astype(np.uint8))
    if points is None:
        return image
    x, y, width, height = cv2.boundingRect(points)
    return image[y : y + height, x : x + width]


def _quality(
    accepted: bool,
    problems: list[str],
    blur_score: float,
    brightness: float,
    coverage: float,
    glare: float = 0.0,
) -> dict:
    guidance = []
    for problem in problems:
        if "blurred" in problem:
            guidance.append("Stabilize the camera and ask the patient to look at a fixed point.")
        elif "dark" in problem:
            guidance.append("Improve illumination or camera alignment and recapture.")
        elif "overexposed" in problem or "glare" in problem:
            guidance.append("Adjust the camera angle to reduce reflection.")
        elif "field" in problem:
            guidance.append("Move closer and centre the retinal field.")
        elif "resolution" in problem:
            guidance.append("Use the original full-resolution capture.")
    return {
        "accepted": accepted,
        "problems": problems,
        "guidance": list(dict.fromkeys(guidance)),
        "metrics": {
            "blur_score": round(blur_score, 1),
            "brightness": round(brightness, 1),
            "field_coverage": round(coverage, 3),
            "glare_fraction": round(glare, 3),
        },
    }
