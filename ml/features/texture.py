"""Texture features using LBP, GLCM, and a small Gabor bank."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return normalized local texture descriptors."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    ellipse = ((xx - width / 2) / (width * 0.35)) ** 2 + ((yy - height / 2) / (height * 0.35)) ** 2
    interior = ellipse <= 1
    ring = (ellipse > 1) & (ellipse <= (0.95 / 0.70) ** 2)
    features: list[float] = []
    names: list[str] = []
    for points, radius in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gray, points, radius, method="uniform")
        bins = points + 2
        for region_name, mask in (("interior", interior), ("ring", ring)):
            histogram, _ = np.histogram(lbp[mask], bins=np.arange(bins + 1), range=(0, bins))
            histogram = histogram.astype(np.float32)
            histogram /= max(float(histogram.sum()), 1.0)
            features.extend(histogram.tolist())
            names.extend([f"texture_lbp{points}_{region_name}_{i}" for i in range(bins)])
    quantized = (gray.astype(np.float32) / 8).astype(np.uint8)
    matrix = graycomatrix(quantized, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        values = graycoprops(matrix, prop).mean(axis=1)
        features.extend(values.tolist())
        names.extend([f"texture_glcm_{prop}_{distance}" for distance in (1, 2)])
    for orientation in range(4):
        for frequency in (0.08, 0.16, 0.32):
            kernel = cv2.getGaborKernel((21, 21), 4.0, orientation * np.pi / 4, 1 / frequency, 0.5, 0, ktype=cv2.CV_32F)
            response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
            features.extend([float(np.mean(np.abs(response))), float(np.std(response))])
            names.extend([f"texture_gabor_{orientation}_{frequency}_{stat}" for stat in ("mean", "std")])
    return np.asarray(features, dtype=np.float32), names