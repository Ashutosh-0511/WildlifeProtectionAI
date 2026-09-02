from __future__ import annotations

import json
import random
from pathlib import Path
from collections import defaultdict


def split_manifest(input_path: str | Path, output_dir: str | Path,
                   train_ratio: float = 0.8, val_ratio: float = 0.1,
                   seed: int = 42) -> dict[str, int]:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    items = data.get("items", [])
    groups = defaultdict(list)
    for item in items:
        key = item.get("clip_id") or item.get("source_video") or item.get("clip_path")
        groups[str(key)].append(item)

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    split_keys = {
        "train": keys[:train_end],
        "val": keys[train_end:val_end],
        "test": keys[val_end:],
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, selected in split_keys.items():
        payload = dict(data)
        payload["split"] = name
        payload["items"] = [item for key in selected for item in groups[key]]
        target = out / f"{name}.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        counts[name] = len(payload["items"])
    (out / "split_metadata.json").write_text(
        json.dumps({"seed": seed, "group_count": n, "counts": counts}, indent=2),
        encoding="utf-8")
    return counts
