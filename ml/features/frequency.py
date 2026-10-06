"""Frequency-domain features for face-swap artifact detection."""
from __future__ import annotations

import cv2
import numpy as np


def extract(image: np.ndarray, bins: int = 64) -> tuple[np.ndarray, list[str]]:
    """Return radial FFT and banded DCT statistics for a BGR face crop."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1])).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray * window))))
    power = spectrum * spectrum
    yy, xx = np.indices(power.shape)
    radius = np.sqrt((xx - power.shape[1] / 2) ** 2 + (yy - power.shape[0] / 2) ** 2)
    max_radius = max(float(radius.max()), 1.0)
    radial = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        low = max_radius * index / bins
        high = max_radius * (index + 1) / bins
        values = power[(radius >= low) & (radius < high)]
        radial[index] = float(values.mean()) if values.size else 0.0
    radial /= max(float(radial.sum()), 1e-8)
    total = max(float(power.sum()), 1e-8)
    high = power[radius >= max_radius * 0.5].sum() / total
    low = power[radius < max_radius * 0.25].sum() / total
    mid = power[(radius >= max_radius * 0.25) & (radius < max_radius * 0.5)].sum() / total
    weights = power / total
    centroid = float((radius * weights).sum() / max_radius)
    flatness = float(np.exp(np.mean(np.log(power + 1e-8))) / (np.mean(power) + 1e-8))
    log_radius = np.log(np.arange(1, bins + 1, dtype=np.float32))
    slope = float(np.polyfit(log_radius, np.log(radial + 1e-8), 1)[0])
    high_values = power[radius >= max_radius * 0.5]
    scalars = [high, mid, low, centroid, flatness, slope, float(high_values.max() / (high_values.mean() + 1e-8)), float(power.std())]
    dct = cv2.dct(gray)
    bands = [dct[:32, :32], dct[32:160, 32:160], dct[160:, 160:]]
    values: list[float] = []
    for band in bands:
        magnitude = np.abs(band)
        values.extend([float(np.mean(magnitude)), float(np.std(magnitude)), float(np.mean(magnitude**2))])
    features = np.asarray(radial.tolist() + scalars + values, dtype=np.float32)
    names = [f"frequency_radial_{i:02d}" for i in range(bins)]
    names += ["frequency_high_ratio", "frequency_mid_ratio", "frequency_low_ratio", "frequency_centroid", "frequency_flatness", "frequency_slope", "frequency_high_peak_mean", "frequency_std"]
    names += [f"frequency_dct_{band}_{stat}" for band in ("low", "mid", "high") for stat in ("mean", "std", "energy")]
    return features, names