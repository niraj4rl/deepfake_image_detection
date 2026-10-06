"""Face detection and deterministic cropping for FaceForensics++ frames."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _largest_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    """Return the largest detected box, or ``None`` when no face exists."""
    return max(boxes, key=lambda box: box[2] * box[3], default=None)


def detect_largest_face(
    image: np.ndarray,
    dnn_net: cv2.dnn_Net | None = None,
    haar: cv2.CascadeClassifier | None = None,
) -> tuple[int, int, int, int] | None:
    """Detect and return the largest face as ``(x, y, width, height)``."""
    height, width = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    if dnn_net is not None:
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        for detection in detections[0, 0]:
            confidence = float(detection[2])
            if confidence < 0.5:
                continue
            left, top, right, bottom = (detection[3:7] * [width, height, width, height]).astype(int)
            x = max(0, left)
            y = max(0, top)
            boxes.append((x, y, min(width, right) - x, min(height, bottom) - y))
    if not boxes and haar is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = [tuple(int(value) for value in box) for box in haar.detectMultiScale(gray, 1.1, 5)]
    return _largest_box([box for box in boxes if box[2] > 0 and box[3] > 0])


def crop_face(image: np.ndarray, box: tuple[int, int, int, int], size: int = 256, margin: float = 1.3) -> np.ndarray:
    """Expand a face box, make it square, clamp it, and resize losslessly."""
    x, y, width, height = box
    center_x = x + width / 2
    center_y = y + height / 2
    side = max(width, height) * margin
    left = max(0, int(round(center_x - side / 2)))
    top = max(0, int(round(center_y - side / 2)))
    right = min(image.shape[1], int(round(center_x + side / 2)))
    bottom = min(image.shape[0], int(round(center_y + side / 2)))
    crop = image[top:bottom, left:right]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)


def load_detectors(model_dir: Path) -> tuple[cv2.dnn_Net | None, cv2.CascadeClassifier]:
    """Load the optional DNN detector and the bundled Haar fallback."""
    prototxt = model_dir / "deploy.prototxt"
    weights = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    dnn_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights)) if prototxt.exists() and weights.exists() else None
    haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    haar = cv2.CascadeClassifier(str(haar_path))
    if haar.empty():
        raise RuntimeError(f"Unable to load Haar face detector: {haar_path}")
    return dnn_net, haar