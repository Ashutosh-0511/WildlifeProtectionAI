from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEFAULT_CLASSES = [
    "RESTING",
    "NORMAL_MOVEMENT",
    "RUNNING",
    "CHASING",
    "AGGRESSIVE_ABNORMAL",
    "UNKNOWN",
]


def _load_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("items", []), payload.get("classes")


class SequenceDataset(Dataset):
    def __init__(self, manifest_path, image_size=224, classes=None):
        self.items, manifest_classes = _load_manifest(manifest_path)
        classes = classes or manifest_classes or DEFAULT_CLASSES
        if isinstance(classes, dict):
            classes = list(classes.keys())
        self.classes = list(classes)
        self.labels = {name: i for i, name in enumerate(self.classes)}
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        frames = []
        for path in item["frames"]:
            with Image.open(path).convert("RGB") as im:
                frames.append(self.transform(im))
        if not frames:
            raise ValueError(f"No frames for sequence {idx}")
        label = str(item["label"])
        if label not in self.labels:
            raise ValueError(
                f"Label {label!r} is not present in manifest classes {self.classes}"
            )
        return torch.stack(frames), self.labels[label]


class ResNetLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=128, layers=1, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.lstm = nn.LSTM(
            self.feature_dim,
            hidden_size,
            layers,
            batch_first=True,
            dropout=0.0 if layers == 1 else 0.2,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        feats = self.encoder(x.reshape(b * t, c, h, w)).flatten(1)
        seq = feats.reshape(b, t, -1)
        out, _ = self.lstm(seq)
        return self.classifier(out[:, -1])


def _run_epoch(model, loader, loss_fn, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0
    ys, preds = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        count += y.size(0)
        ys.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())

    return total_loss / max(count, 1), correct / max(count, 1), ys, preds


def macro_f1(y_true, y_pred, n):
    scores = []
    for k in range(n):
        tp = sum(a == k and b == k for a, b in zip(y_true, y_pred))
        fp = sum(a != k and b == k for a, b in zip(y_true, y_pred))
        fn = sum(a == k and b != k for a, b in zip(y_true, y_pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(scores) / n


def main(train_manifest, val_manifest, output, epochs=5, batch_size=2,
         lr=1e-4, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    train_items, train_classes = _load_manifest(train_manifest)
    val_items, val_classes = _load_manifest(val_manifest)
    if not train_items or not val_items:
        raise RuntimeError("Train/validation manifests must contain items.")

    classes = train_classes or val_classes or DEFAULT_CLASSES
    if isinstance(classes, dict):
        classes = list(classes.keys())
    classes = list(classes)

    unknown = {
        str(x["label"]) for x in train_items + val_items
        if str(x["label"]) not in classes
    }
    if unknown:
        raise RuntimeError(f"Manifest contains unknown labels: {sorted(unknown)}")

    train_ds = SequenceDataset(train_manifest, classes=classes)
    val_ds = SequenceDataset(val_manifest, classes=classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResNetLSTM(num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        tl, ta, _, _ = _run_epoch(model, train_loader, loss_fn, optimizer, device)
        with torch.no_grad():
            vl, va, vy, vp = _run_epoch(model, val_loader, loss_fn, None, device)
        vf1 = macro_f1(vy, vp, len(classes))
        row = {
            "epoch": epoch,
            "train_loss": tl,
            "train_accuracy": ta,
            "val_loss": vl,
            "val_accuracy": va,
            "val_macro_f1": vf1,
        }
        history.append(row)
        print(row)

        if vf1 > best_f1:
            best_f1 = vf1
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "model": "resnet18_lstm",
                "seed": seed,
                "best_val_macro_f1": best_f1,
            }, output)

    Path(str(output) + ".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(
        f"BEHAVIOR_TRAIN_OK: best_val_macro_f1={best_f1:.4f} "
        f"checkpoint={output}"
    )
