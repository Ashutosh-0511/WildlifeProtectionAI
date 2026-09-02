from __future__ import annotations

import argparse
from pathlib import Path

from ml.species.speciesnet import SpeciesNetAdapter


def main() -> None:
    p = argparse.ArgumentParser(description="Run SpeciesNet on a folder of animal crops.")
    p.add_argument("--folder", required=True, type=Path)
    p.add_argument("--output", default="data/outputs/speciesnet/predictions.json", type=Path)
    args = p.parse_args()
    if not args.folder.exists():
        raise SystemExit(f"Crop folder not found: {args.folder}")
    adapter = SpeciesNetAdapter(country="IND")
    result = adapter.classify_folder(args.folder, args.output)
    print(f"SpeciesNet predictions: {len(result.get('predictions', []))}")
    print(f"Output: {args.output.resolve()}")
    print("SPECIESNET_INFERENCE_OK")


if __name__ == "__main__":
    main()
