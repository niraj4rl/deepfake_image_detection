"""Build the real-data face manifest from sampled FF++ frames and split JSONs."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import cv2

from ml.config import load_config, root_path
from ml.data.face_crop import crop_face, detect_largest_face, load_detectors

LOGGER = logging.getLogger(__name__)
FIELDS = ["image_path", "label", "manipulation", "video_id", "group_id", "frame_idx", "split"]


def _split_ids(path: Path, limit: int) -> set[str]:
    """Read an official FF++ split JSON and return its selected video IDs."""
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    ids: list[str] = []
    for value in values:
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            identifier = Path(str(candidate)).stem
            if identifier not in ids:
                ids.append(identifier)
    return set(ids[:limit])


def _split_map(split_root: Path, limits: dict[str, int]) -> dict[str, str]:
    """Map selected target video IDs to their official split."""
    mapping: dict[str, str] = {}
    for split, limit in limits.items():
        split_path = split_root / f"{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing official split file: {split_path}")
        for video_id in _split_ids(split_path, limit):
            if video_id in mapping:
                raise ValueError(f"Video ID appears in multiple splits: {video_id}")
            mapping[video_id] = split
    return mapping


def _target_id(video_id: str, manipulation: str) -> str:
    """Return the real target ID represented by a real or fake video name."""
    return video_id if manipulation == "real" else video_id.split("_", 1)[0]


def main() -> None:
    """Crop sampled real FF++ faces and write ``data/manifest.csv``."""
    config = load_config()
    frames_root = root_path("data/frames")
    faces_root = root_path(config["paths"]["faces"])
    if not frames_root.exists():
        raise FileNotFoundError(f"Missing sampled frames: {frames_root}; run ml.data.extract_frames first")
    limits = config["data"]["subset"]
    split_map = _split_map(root_path("ml/data/splits"), limits)
    dnn_net, haar = load_detectors(root_path("models/face_detector"))
    rows: list[dict[str, str | int]] = []
    for manipulation in ["real", *config["data"]["manipulations"]]:
        source_root = frames_root / manipulation
        if not source_root.exists():
            LOGGER.warning("No %s frames found; continuing with available real data", manipulation)
            continue
        for video_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            video_id = video_dir.name
            group_id = _target_id(video_id, manipulation)
            split = split_map.get(group_id)
            if split is None:
                continue
            for frame_path in sorted(video_dir.glob("*.jpg")):
                image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if image is None:
                    LOGGER.warning("Skipping unreadable frame: %s", frame_path)
                    continue
                box = detect_largest_face(image, dnn_net, haar)
                if box is None:
                    LOGGER.warning("No face detected: %s", frame_path)
                    continue
                face = crop_face(image, box, config["data"]["face_size"], config["data"]["face_margin"])
                output = faces_root / split / manipulation / f"{video_id}_{frame_path.stem.split('_')[-1]}.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output), face):
                    raise OSError(f"Could not write face crop: {output}")
                rows.append({
                    "image_path": output.relative_to(root_path("." )).as_posix(),
                    "label": 0 if manipulation == "real" else 1,
                    "manipulation": manipulation,
                    "video_id": video_id,
                    "group_id": group_id,
                    "frame_idx": frame_path.stem.split("_")[-1],
                    "split": split,
                })
    manifest = root_path(config["paths"]["manifest"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} real face crops to {manifest}.")


if __name__ == "__main__":
    main()