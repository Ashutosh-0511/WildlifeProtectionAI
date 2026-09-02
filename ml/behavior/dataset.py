from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights


class TrackSequenceDataset(Dataset):
    """Loads fixed-length image sequences from a JSON manifest."""
    def __init__(self, manifest: str | Path, sequence_length: int = 8, transform=None):
        self.manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.sequence_length = sequence_length
        self.items = self.manifest.get("items", self.manifest if isinstance(self.manifest, list) else [])
        self.transform = transform or ResNet18_Weights.DEFAULT.transforms()
        self.labels = sorted({str(x["label"]) for x in self.items})
        self.label_to_id = {x: i for i, x in enumerate(self.labels)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        frames = [Path(x) for x in item["frames"]]
        if not frames:
            raise ValueError(f"No frames for sequence {index}")
        if len(frames) >= self.sequence_length:
            idx = torch.linspace(0, len(frames) - 1, self.sequence_length).round().long().tolist()
            frames = [frames[i] for i in idx]
        else:
            frames = frames + [frames[-1]] * (self.sequence_length - len(frames))
        images = [self.transform(Image.open(p).convert("RGB")) for p in frames]
        return torch.stack(images), torch.tensor(self.label_to_id[str(item["label"])], dtype=torch.long)
