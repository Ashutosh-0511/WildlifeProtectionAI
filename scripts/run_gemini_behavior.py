from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.behavior.gemini_video import analyze_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze wildlife video behaviour with Gemini")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    output = args.output or Path("data/outputs/gemini") / f"{args.input.stem}.json"
    result = analyze_video(args.input, output_path=output, model=args.model)
    print(json.dumps(result, indent=2))
    print(f"GEMINI_BEHAVIOR_OK: {output}")


if __name__ == "__main__":
    main()
