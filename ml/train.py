"""Train the CPU-only classical detector from the real feature cache."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

from ml.config import load_config, root_path


def main() -> None:
    """Fit and save the real-data SVM pipeline with a held-out test split."""
    config = load_config()
    cache_path = root_path(config["paths"]["features"])
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing real feature cache: {cache_path}; run feature extraction first")
    cache = np.load(cache_path, allow_pickle=False)
    X = cache["X"]
    y = cache["y"]
    splits = cache["split"]
    if len(X) == 0 or set(splits.tolist()) - {"train", "val", "test"}:
        raise ValueError("Feature cache does not contain the expected real train/val/test splits")
    train_mask = splits == "train"
    if train_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2:
        raise ValueError("Real training data must contain both real and fake classes")
    model = Pipeline([
        ("variance", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("classifier", SVC(C=2.0, kernel="rbf", probability=True, random_state=config["seed"])),
    ])
    model.fit(X[train_mask], y[train_mask])
    model_dir = root_path(config["paths"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "best_model.joblib")
    metadata = {
        "model_name": "SVM-RBF",
        "feature_count": int(X.shape[1]),
        "feature_names": cache["feature_names"].tolist(),
        "dataset_counts": {split: int(np.sum(splits == split)) for split in ("train", "val", "test")},
        "classes": {"real": 0, "fake": 1},
        "training_date": datetime.now(timezone.utc).isoformat(),
        "test_is_held_out": True,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved real-data model to {model_dir / 'best_model.joblib'}.")


if __name__ == "__main__":
    main()