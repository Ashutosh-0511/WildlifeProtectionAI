from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor


class VideoMAEBehaviorClassifier:
    """Local VideoMAE behaviour classifier for the wildlife V1 pipeline."""

    LABELS = (
        "Standing",
        "Lying",
        "Foraging/Grazing",
        "Drinking",
        "Ruminating",
        "Grooming",
        "Other",
    )
    BASE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"

    def __init__(
        self,
        checkpoint_path: str | Path = "models/behavior/videomae/videomae_combined_v1.pt",
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"VideoMAE checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device)
        if self.device.type != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {device!r}, but CUDA is unavailable.")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError("Invalid checkpoint: expected dict containing 'model_state'.")

        config = VideoMAEConfig.from_pretrained(self.BASE_MODEL)
        config.num_labels = len(self.LABELS)
        config.id2label = {i: label for i, label in enumerate(self.LABELS)}
        config.label2id = {label: i for i, label in enumerate(self.LABELS)}

        self.processor = VideoMAEImageProcessor.from_pretrained(self.BASE_MODEL)
        self.model = VideoMAEForVideoClassification(config)
        state = dict(checkpoint["model_state"])
        # Original VideoMAE checkpoints store Q/V biases separately; current
        # Transformers expects separate query/key/value bias tensors.
        for i in range(12):
            prefix = f"videomae.encoder.layer.{i}.attention.attention"
            q = state.pop(f"{prefix}.q_bias", None)
            v = state.pop(f"{prefix}.v_bias", None)
            if q is not None and v is not None:
                state[f"{prefix}.query.bias"] = q
                state[f"{prefix}.key.bias"] = torch.zeros_like(q)
                state[f"{prefix}.value.bias"] = v
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        self.epoch = checkpoint.get("epoch")
        self.val_macro_f1 = checkpoint.get("val_macro_f1")

    @property
    def num_frames(self) -> int:
        return int(getattr(self.model.config, "num_frames", 16))

    @torch.inference_mode()
    def predict(self, frames: Sequence[np.ndarray | Image.Image]) -> dict:
        if not frames:
            raise ValueError("No frames supplied to VideoMAE.")

        clip = list(frames)
        target = self.num_frames
        if len(clip) < target:
            clip += [clip[-1]] * (target - len(clip))
        elif len(clip) > target:
            indices = np.linspace(0, len(clip) - 1, target).round().astype(int)
            clip = [clip[i] for i in indices]

        images = []
        for frame in clip:
            if isinstance(frame, Image.Image):
                images.append(frame.convert("RGB"))
            elif isinstance(frame, np.ndarray):
                images.append(Image.fromarray(frame).convert("RGB"))
            else:
                raise TypeError("Frames must be NumPy arrays or PIL Images.")

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        confidence, index = probabilities.max(dim=-1)

        return {
            "behaviour": self.LABELS[int(index.item())],
            "confidence": float(confidence.item()),
            "probabilities": {
                label: float(probabilities[i].item())
                for i, label in enumerate(self.LABELS)
            },
            "model_version": "VideoMAE-CattleVision-v1",
            "checkpoint_epoch": self.epoch,
            "checkpoint_val_macro_f1": self.val_macro_f1,
            "frames": len(clip),
        }

    def predict_paths(self, paths: Iterable[str | Path]) -> dict:
        frames = [np.asarray(Image.open(Path(p)).convert("RGB")) for p in paths]
        return self.predict(frames)
