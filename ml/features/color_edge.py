"""Color distribution and boundary features."""
from __future__ import annotations

import cv2
import numpy as np


def extract(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return YCbCr/HSV statistics, chrominance histograms, and edge features."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    features: list[float] = []
    names: list[str] = []
    for space, values in (("ycrcb", ycrcb), ("hsv", hsv)):
        for channel, value in enumerate(cv2.split(values)):
            features.extend([float(value.mean()), float(value.std()), float(np.mean((value - value.mean()) ** 3) / (value.std() ** 3 + 1e-8))])
            names.extend([f"color_{space}_{channel}_{stat}" for stat in ("mean", "std", "skew")])
    for channel, value in (("cb", ycrcb[:, :, 1]), ("cr", ycrcb[:, :, 2])):
        histogram, _ = np.histogram(value, bins=16, range=(0, 256))
        histogram = histogram.astype(np.float32) / max(float(histogram.sum()), 1.0)
        features.extend(histogram.tolist())
        names.extend([f"color_{channel}_hist_{i}" for i in range(16)])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    edges = cv2.Canny(gray, max(0, int(0.67 * median)), min(255, int(1.33 * median)))
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt(((xx - width / 2) / width) ** 2 + ((yy - height / 2) / height) ** 2)
    interior = radius <= 0.35
    ring = (radius > 0.35) & (radius <= 0.48)
    density = [float(np.mean(edges > 0)), float(np.mean(edges[ring] > 0)), float(np.mean(edges[interior] > 0))]
    features.extend(density + [density[1] / max(density[2], 1e-8)])
    names.extend(["edge_density", "edge_ring_density", "edge_interior_density", "edge_ring_interior_ratio"])
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    angle = (np.arctan2(gy, gx) + np.pi) % np.pi
    features.extend([float(magnitude.mean()), float(magnitude.std()), float(np.percentile(magnitude, 95))])
    names.extend(["edge_gradient_mean", "edge_gradient_std", "edge_gradient_p95"])
    histogram, _ = np.histogram(angle, bins=8, range=(0, np.pi), weights=magnitude)
    histogram = histogram.astype(np.float32) / max(float(histogram.sum()), 1.0)
    features.extend(histogram.tolist())
    names.extend([f"edge_orientation_{i}" for i in range(8)])
    return np.asarray(features, dtype=np.float32), names