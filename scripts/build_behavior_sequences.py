from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.behavior.sequence_builder import build_sequences, write_manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Extract fixed-length behaviour sequences from labelled videos.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--output-root", default="data/processed/behavior_sequences")
    p.add_argument("--manifest", default="data/processed/behavior_sequences_manifest.json")
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    args = p.parse_args()

    items = build_sequences(args.input, args.label, args.output_root,
                            args.sequence_length, args.stride, args.split)
    existing = []
    if Path(args.manifest).exists():
        existing = json.loads(Path(args.manifest).read_text(encoding="utf-8")).get("items", [])
    write_manifest(existing + items, args.manifest, "project_behavior_sequences")
    print(f"BEHAVIOR_SEQUENCE_BUILD_OK: {len(items)} sequences")


if __name__ == "__main__":
    main()
