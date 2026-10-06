"""Small, deterministic classical feature extractor used by both model paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from ml.config import load_config, root_path
from ml.features.color_edge import extract as extract_color_edge
from ml.features.ela_noise import extract as extract_ela_noise
from ml.features.frequency import extract as extract_frequency
from ml.features.texture import extract as extract_texture

def _stats(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    values = values.astype(np.float32)
    data = [float(np.mean(values)), float(np.std(values)), float(np.percentile(values, 95))]
    return data, [f"{prefix}_{name}" for name in ("mean", "std", "p95")]


def extract_features(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract finite classical DIP features from a BGR uint8 face crop."""
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty BGR array")
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    groups = [extract_frequency(image), extract_ela_noise(image), extract_texture(image), extract_color_edge(image)]
    vector = np.nan_to_num(np.concatenate([values for values, _ in groups]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    names = [name for _, group_names in groups for name in group_names]
    return vector, names


def extract_image(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read an image path and extract its features."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return extract_features(image)


FEATURE_NAMES = extract_features(np.zeros((256, 256, 3), dtype=np.uint8))[1]


def _extract_manifest_row(path: str) -> np.ndarray:
    """Extract one manifest image for use by the parallel worker pool."""
    if "data/dummy" in path.replace("\\", "/"):
        raise ValueError("Synthetic data is not permitted in the real feature cache")
    return extract_image(root_path(path))[0]


def build_cache() -> Path:
    """Extract real manifest images in parallel and write the configured NPZ cache."""
    config = load_config()
    manifest_path = root_path(config["paths"]["manifest"])
    cache_path = root_path(config["paths"]["features"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing real manifest: {manifest_path}; run ml.data.build_manifest first")
    manifest = pd.read_csv(manifest_path)
    required = {"image_path", "label", "group_id", "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("The real manifest contains no face crops")
    for image_path in manifest["image_path"]:
        if not root_path(str(image_path)).exists():
            raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    features = Parallel(n_jobs=-1)(delayed(_extract_manifest_row)(str(path)) for path in manifest["image_path"])
    matrix = np.asarray(features, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=matrix,
        y=manifest["label"].to_numpy(dtype=np.int64),
        groups=manifest["group_id"].astype(str).to_numpy(),
        split=manifest["split"].astype(str).to_numpy(),
        feature_names=np.asarray(FEATURE_NAMES),
        image_paths=manifest["image_path"].astype(str).to_numpy(),
        config_hash=np.asarray(config_hash),
    )
    print(f"Cached {len(matrix)} real images with {matrix.shape[1]} features at {cache_path}.")
    return cache_path


if __name__ == "__main__":
    build_cache()
