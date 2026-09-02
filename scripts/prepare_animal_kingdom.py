from __future__ import annotations

import argparse

from ml.behavior.animal_kingdom import AnimalKingdomAdapter


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Animal Kingdom sequences for ResNet18-LSTM.")
    p.add_argument("--annotations", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--output", default="data/processed/animal_kingdom_manifest.json")
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--include-unknown", action="store_true")
    args = p.parse_args()

    adapter = AnimalKingdomAdapter(args.annotations, args.video_root)
    count = adapter.build_sequences(
        args.output,
        sequence_length=args.sequence_length,
        include_unknown=args.include_unknown,
    )
    print(f"ANIMAL_KINGDOM_SEQUENCES_OK: {count} sequences -> {args.output}.parent manifests")


if __name__ == "__main__":
    main()
