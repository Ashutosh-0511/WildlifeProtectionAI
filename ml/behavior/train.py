from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml.behavior.model import ResNetLSTM


class SequenceDataset(Dataset):
    def __init__(self, manifest: str, class_names: list[str], sequence_length: int = 8):
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.rows = payload.get("items", payload.get("rows", payload if isinstance(payload, list) else []))
        self.class_to_id = {name: i for i, name in enumerate(class_names)}
        self.sequence_length = sequence_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        paths = row.get("frame_paths") or row.get("frames") or []
        if len(paths) < self.sequence_length:
            raise ValueError(f"Row {idx} has {len(paths)} frames; need {self.sequence_length}")
        if len(paths) > self.sequence_length:
            step = (len(paths) - 1) / (self.sequence_length - 1)
            paths = [paths[round(i * step)] for i in range(self.sequence_length)]
        label = row.get("label")
        if label not in self.class_to_id:
            raise ValueError(f"Unknown label {label!r} in row {idx}")
        images = [self.transform(Image.open(p).convert("RGB")) for p in paths]
        return torch.stack(images), self.class_to_id[label]


def train(manifest: str, classes: list[str], epochs: int, output: str, sequence_length: int):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")
    ds = SequenceDataset(manifest, classes, sequence_length)
    if len(ds) == 0:
        raise RuntimeError("No training sequences found in manifest.")
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    model = ResNetLSTM(len(classes), pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch={epoch+1}/{epochs} loss={total/max(1,len(loader)):.4f}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": classes, "sequence_length": sequence_length}, output)
    print(f"BEHAVIOUR_MODEL_TRAINED: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--classes", nargs="+", default=["RESTING", "NORMAL_MOVEMENT", "RUNNING", "UNKNOWN"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--output", default="models/behavior/resnet18_lstm.pt")
    a = p.parse_args()
    train(a.manifest, a.classes, a.epochs, a.output, a.sequence_length)
