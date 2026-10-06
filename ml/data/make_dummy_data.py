"""Generate a clearly synthetic local dataset for development smoke tests."""
from pathlib import Path
import cv2
import numpy as np

from ml.config import ROOT


def main() -> None:
    """Write a small synthetic real/fake image set and manifest."""
    output = ROOT / "data" / "dummy"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    for label, name in ((0, "real"), (1, "Deepfakes")):
        for index in range(20):
            image = rng.integers(20, 235, (256, 256, 3), dtype=np.uint8)
            if label:
                cv2.rectangle(image, (45, 45), (210, 210), (0, 0, 255), 3)
                image = cv2.GaussianBlur(image, (9, 9), 0)
            path = output / f"{name}_{index:03d}.png"
            cv2.imwrite(str(path), image)
            rows.append(f"{path.as_posix()},{label},{name},video_{index},video_{index},0,train")
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "manifest.csv").write_text(
        "image_path,label,manipulation,video_id,group_id,frame_idx,split\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    print("WARNING: dummy data generated; never use these results in the project report.")


if __name__ == "__main__":
    main()
