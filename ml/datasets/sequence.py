from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class SequenceSample:
    track_id: str
    frame_paths: tuple[str, ...]
    label: str


def build_sequences(frame_dir: str | Path, track_id: str, label: str, sequence_length: int = 16, stride: int = 4) -> Iterator[SequenceSample]:
    """Create deterministic fixed-length temporal samples from extracted crops.

    Expected filenames: <track_id>_<frame_index>.jpg
    """
    root = Path(frame_dir)
    paths = sorted(root.glob(f"{track_id}_*.jpg"))
    for start in range(0, max(0, len(paths) - sequence_length + 1), max(1, stride)):
        window = paths[start:start + sequence_length]
        if len(window) == sequence_length:
            yield SequenceSample(track_id, tuple(map(str, window)), label)
