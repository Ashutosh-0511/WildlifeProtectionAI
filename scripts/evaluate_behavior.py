from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from ml.behavior.train_lstm import ResNetLSTM, SequenceDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--output", default="data/outputs/behavior/evaluation.json")
    a = p.parse_args()

    payload = torch.load(a.checkpoint, map_location="cpu")
    classes = payload.get("classes")
    if not classes:
        raise RuntimeError("Checkpoint does not contain its class vocabulary.")

    ds = SequenceDataset(a.manifest, classes=classes)
    if len(ds) == 0:
        raise RuntimeError("Evaluation manifest is empty.")

    model = ResNetLSTM(num_classes=len(classes))
    model.load_state_dict(payload["state_dict"])
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            y_true.extend(y.tolist())
            y_pred.extend(logits.argmax(1).tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "model": str(Path(a.checkpoint)),
        "manifest": str(Path(a.manifest)),
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1),
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(classes))
        )).tolist(),
        "classes": classes,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "classes": classes,
    }, indent=2))
    print(f"BEHAVIOR_EVALUATION_OK: {out}")


if __name__ == "__main__":
    main()
