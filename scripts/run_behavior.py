from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.behavior import BehaviorMapper, VideoMAEBehaviorClassifier

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VideoMAE behaviour inference.")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--checkpoint", default="models/behavior/videomae/videomae_combined_v1.pt")
    parser.add_argument("--output", default="data/processed/behavior/prediction.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    folder = Path(args.folder)
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    if not paths:
        raise RuntimeError(f"No image frames found in {folder}")

    prediction = BehaviorMapper.enrich(
        VideoMAEBehaviorClassifier(args.checkpoint, args.device).predict_paths(paths)
    )
    prediction["input_folder"] = str(folder)
    prediction["input_frames"] = len(paths)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    print(json.dumps(prediction, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
