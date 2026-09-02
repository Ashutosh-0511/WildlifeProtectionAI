from __future__ import annotations

import argparse
from ml.behavior.train_lstm import main

p = argparse.ArgumentParser(description="Train the ResNet18 + LSTM behaviour model on sequence manifests.")
p.add_argument("--train", default="data/processed/behavior_splits/train.json")
p.add_argument("--val", default="data/processed/behavior_splits/val.json")
p.add_argument("--output", default="data/models/behavior/resnet18_lstm.pt")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch-size", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
main(a.train, a.val, a.output, a.epochs, a.batch_size, a.lr, a.seed)
