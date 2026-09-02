from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2


def _sample_indices(start: int, end: int, length: int) -> list[int]:
    if end <= start:
        return [start] * length
    if end - start + 1 <= length:
        values = list(range(start, end + 1))
        return values + [values[-1]] * (length - len(values))
    step = (end - start) / (length - 1)
    return [round(start + i * step) for i in range(length)]


def build_sequences(video: str | Path, label: str, output_root: str | Path,
                    sequence_length: int = 8, stride: int = 4,
                    split: str = "train", clip_id: str | None = None) -> list[dict]:
    video = str(video)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    output_root = Path(output_root)
    clip_name = clip_id or Path(video).stem
    items = []
    for start in range(0, max(1, total), stride):
        end = min(total - 1, start + sequence_length - 1)
        if end < start:
            break
        indices = _sample_indices(start, end, sequence_length)
        seq_dir = output_root / split / label / clip_name / f"{start:08d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, frame_no in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                break
            path = seq_dir / f"{i:04d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(str(path.resolve()))
        if len(paths) == sequence_length:
            items.append({"frames": paths, "label": label, "clip_id": clip_name,
                          "source_video": str(Path(video).resolve()), "start_frame": start,
                          "end_frame": end, "fps": fps})
    cap.release()
    return items


def write_manifest(items: Iterable[dict], output: str | Path, dataset: str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": dataset, "items": list(items)}, indent=2), encoding="utf-8")
