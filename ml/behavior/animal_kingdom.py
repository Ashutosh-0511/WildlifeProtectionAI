from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2


PRIORITY = [
    ("AGGRESSIVE_ABNORMAL", ["attack", "fight", "aggress", "bite", "strike"]),
    ("CHASING", ["chase", "pursu", "hunt"]),
    ("RUNNING", ["run", "running", "gallop", "flee", "escape"]),
    ("NORMAL_MOVEMENT", ["walk", "walking", "move", "moving", "locomot", "crawl", "swim", "fly", "climb"]),
    ("RESTING", ["rest", "sleep", "sit", "lie", "stand", "standing"]),
]


class AnimalKingdomAdapter:
    """Convert Animal Kingdom Charades-style CSV rows into project sequences.

    The source uses multi-label actions; the project baseline is single-label.
    The mapping is therefore explicit, deterministic and preserved in each item.
    """

    def __init__(self, annotation_csv: str | Path, video_root: str | Path | None = None):
        self.annotation_csv = Path(annotation_csv)
        self.video_root = Path(video_root) if video_root else None

    @staticmethod
    def _labels(value: str) -> list[str]:
        value = (value or "").strip()
        if not value:
            return []
        for sep in (";", ","):
            if sep in value:
                return [x.strip() for x in value.split(sep) if x.strip()]
        return [value]

    @staticmethod
    def map_behaviour(labels: list[str]) -> str | None:
        text = " ".join(labels).lower().replace("_", " ")
        for target, keywords in PRIORITY:
            if any(k in text for k in keywords):
                return target
        return None

    def read(self) -> list[dict[str, Any]]:
        with self.annotation_csv.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        return [{
            "clip_id": row.get("clip_id", ""),
            "clip_number": row.get("clip_number", ""),
            "frame_number": row.get("frame_number", ""),
            "clip_path": row.get("clip_path") or row.get("video_path") or row.get("path") or "",
            "action_labels": self._labels(row.get("action_labels", "")),
        } for row in rows]

    def _resolve_video(self, clip_path: str) -> Path:
        p = Path(clip_path)
        candidates = [p]
        if self.video_root:
            candidates += [self.video_root / p, self.video_root / p.name]
            candidates += list(self.video_root.rglob(p.name))
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"Video not found: {clip_path}")

    @staticmethod
    def _extract(video: Path, center: int, length: int, cache: Path, key: str) -> list[str]:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        end = min(total - 1, center + length // 2)
        start = max(0, end - length + 1)
        paths = []
        cache.mkdir(parents=True, exist_ok=True)
        for n in range(start, end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = cap.read()
            if not ok:
                continue
            path = cache / f"{key}_f{n:06d}.jpg"
            if not path.exists():
                cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            paths.append(str(path))
        cap.release()
        return paths

    def build_sequences(
        self, output: str | Path, sequence_length: int = 8, include_unknown: bool = False
    ) -> int:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        cache = out.parent / "frames"
        buckets = {"train": [], "val": [], "test": []}
        skipped = 0

        for row in self.read():
            label = self.map_behaviour(row["action_labels"])
            if label is None and not include_unknown:
                skipped += 1
                continue
            label = label or "UNKNOWN"
            try:
                center = int(float(row["frame_number"]))
                video = self._resolve_video(row["clip_path"])
                key = hashlib.sha1(f"{row['clip_id']}|{center}".encode()).hexdigest()[:16]
                frames = self._extract(video, center, sequence_length, cache, key)
            except Exception as exc:
                print(f"SKIP {row['clip_id']}: {exc}")
                skipped += 1
                continue
            if len(frames) != sequence_length:
                skipped += 1
                continue

            clip_id = str(row["clip_id"])
            bucket_id = int(hashlib.sha1(clip_id.encode()).hexdigest()[:8], 16) % 100
            split = "train" if bucket_id < 80 else "val" if bucket_id < 90 else "test"
            buckets[split].append({
                "clip_id": clip_id,
                "frame_number": center,
                "source_action_labels": row["action_labels"],
                "label": label,
                "frames": frames,
            })

        for split, items in buckets.items():
            (out.parent / f"{split}.json").write_text(
                json.dumps({"items": items}, indent=2), encoding="utf-8"
            )
        summary = {
            "dataset": "Animal Kingdom",
            "sequence_length": sequence_length,
            "counts": {k: len(v) for k, v in buckets.items()},
            "skipped": skipped,
            "mapping": PRIORITY,
        }
        (out.parent / "manifest_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return sum(map(len, buckets.values()))

    def to_manifest(self, output: str | Path) -> int:
        rows = self.read()
        payload = {
            "dataset": "Animal Kingdom",
            "task": "action_recognition",
            "items": rows,
            "source_annotation": str(self.annotation_csv.resolve()),
        }
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return len(rows)
