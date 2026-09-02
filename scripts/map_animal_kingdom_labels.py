from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.behavior.ontology import map_actions


def main() -> None:
    p = argparse.ArgumentParser(description="Map preserved Animal Kingdom action labels to project behaviour classes.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/processed/animal_kingdom_behavior_manifest.json")
    args = p.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    items = []
    for item in source.get("items", []):
        row = dict(item)
        row["project_behavior"] = map_actions(item.get("action_labels", []))
        items.append(row)

    payload = dict(source)
    payload["ontology"] = ["RESTING", "NORMAL_MOVEMENT", "RUNNING", "CHASING", "AGGRESSIVE_ABNORMAL", "UNKNOWN"]
    payload["items"] = items

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ANIMAL_KINGDOM_MAPPING_OK: {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
