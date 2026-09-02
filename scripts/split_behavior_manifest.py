from __future__ import annotations

import argparse
from ml.behavior.split import split_manifest

p = argparse.ArgumentParser(description="Split behaviour sequences by source clip to prevent temporal leakage.")
p.add_argument("--input", required=True)
p.add_argument("--output-dir", default="data/processed/behavior_splits")
p.add_argument("--train-ratio", type=float, default=0.8)
p.add_argument("--val-ratio", type=float, default=0.1)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
if a.train_ratio <= 0 or a.val_ratio < 0 or a.train_ratio + a.val_ratio >= 1:
    raise SystemExit("train-ratio + val-ratio must be < 1")
print(split_manifest(a.input, a.output_dir, a.train_ratio, a.val_ratio, a.seed))
print("BEHAVIOR_SPLIT_OK")
