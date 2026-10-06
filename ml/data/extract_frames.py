"""Sample deterministic frames from real FaceForensics++ c23 videos."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

from ml.config import load_config, root_path

LOGGER = logging.getLogger(__name__)


def sample_video(video_path: Path, output_dir: Path, frames_per_video: int) -> int:
    """Extract evenly spaced JPEG frames from one video and return the count."""
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        LOGGER.warning("Skipping video with no readable frames: %s", video_path)
        return 0
    indices = sorted(set(round(index) for index in __import__("numpy").linspace(0, frame_count - 1, frames_per_video)))
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if success:
            target = output_dir / f"{video_path.stem}_{frame_index:06d}.jpg"
            if cv2.imwrite(str(target), frame):
                written += 1
    capture.release()
    return written


def main() -> None:
    """Extract frames for every configured real and manipulated video."""
    config = load_config()
    raw_root = root_path(config["paths"]["raw_data"])
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing real dataset directory: {raw_root}")
    output_root = root_path("data/frames")
    total = 0
    real_root = raw_root / "original_sequences" / "youtube" / config["data"]["quality"] / "videos"
    for video_path in sorted(real_root.glob("*.mp4")):
        total += sample_video(video_path, output_root / "real" / video_path.stem, config["data"]["frames_per_video"])
    for manipulation in config["data"]["manipulations"]:
        video_root = raw_root / "manipulated_sequences" / manipulation / config["data"]["quality"] / "videos"
        for video_path in sorted(video_root.glob("*.mp4")):
            total += sample_video(video_path, output_root / manipulation / video_path.stem, config["data"]["frames_per_video"])
    print(f"Extracted {total} real-data frames into {output_root}.")


if __name__ == "__main__":
    main()