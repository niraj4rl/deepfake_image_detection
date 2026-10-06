"""Error-level and residual-noise features."""
from __future__ import annotations

import cv2
import numpy as np


def _summary(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    values = values.astype(np.float32)
    stats = [float(values.mean()), float(values.std()), float(values.max()), float(np.percentile(values, 95))]
    return stats, [f"{prefix}_{name}" for name in ("mean", "std", "max", "p95")]


def extract(image: np.ndarray, qualities: tuple[int, ...] = (90, 75)) -> tuple[np.ndarray, list[str]]:
    """Return JPEG ELA and fixed-filter noise statistics."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    features: list[float] = []
    names: list[str] = []
    for quality in qualities:
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if success else image
        ela = cv2.absdiff(image, recompressed).astype(np.float32)
        ela_gray = cv2.cvtColor(ela.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        values, value_names = _summary(ela_gray, f"ela_q{quality}")
        features.extend(values + [float(channel.mean()) for channel in cv2.split(ela)])
        names.extend(value_names + [f"ela_q{quality}_{channel}_mean" for channel in ("b", "g", "r")])
        for row in np.array_split(ela_gray, 4, axis=0):
            for block in np.array_split(row, 4, axis=1):
                features.append(float(block.mean()))
                names.append(f"ela_q{quality}_grid_{len(names)}")
    residuals = [gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)]
    kernels = [np.array([[1, -1]], dtype=np.float32), np.array([[1, -2, 1]], dtype=np.float32), np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)]
    residuals.extend([cv2.filter2D(gray, cv2.CV_32F, kernel) for kernel in kernels])
    for index, residual in enumerate(residuals):
        clipped = np.clip(residual, -2, 2)
        stats, stat_names = _summary(np.abs(clipped), f"noise_{index}")
        features.extend(stats + [float(np.var(block)) for row in np.array_split(clipped, 2, axis=0) for block in np.array_split(row, 2, axis=1)])
        names.extend(stat_names + [f"noise_{index}_grid_{grid}" for grid in range(4)])
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    features.append(float(laplacian.var()))
    names.append("noise_laplacian_variance")
    return np.asarray(features, dtype=np.float32), names