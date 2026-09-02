from __future__ import annotations

from pathlib import Path
from typing import Iterator
import cv2


def read_video(path: str | Path, sample_every: int = 1) -> Iterator[tuple[int, float, object]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % max(1, sample_every) == 0:
                yield frame_index, frame_index / fps, frame
            frame_index += 1
    finally:
        cap.release()
