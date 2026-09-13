from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


KINETICS_LABELS_URL = (
    "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
)
X3D_REPO = "facebookresearch/pytorchvideo"

# Kinetics-400 contains many actions that are irrelevant to wildlife behavior.
# Only map labels that carry a useful animal-motion/action signal. Everything
# else is deliberately reported as UNKNOWN rather than inventing a behavior.
_BEHAVIOUR_ALIASES = {
    "running": "RUNNING",
    "running on treadmill": "RUNNING",
    "jogging": "RUNNING",
    "sprinting": "RUNNING",
    "trotting": "RUNNING",
    "walking": "NORMAL_MOVEMENT",
    "walking the dog": "NORMAL_MOVEMENT",
    "standing": "STATIONARY",
    "waiting": "STATIONARY",
    "lying": "RESTING",
    "laying": "RESTING",
    "sleeping": "RESTING",
    "resting": "RESTING",
    "grazing": "FEEDING",
    "feeding": "FEEDING",
    "feeding animals": "FEEDING",
    "feeding birds": "FEEDING",
    "eating": "FEEDING",
    "drinking": "DRINKING",
    "grooming": "GROOMING",
    "bathing": "GROOMING",
    "swimming": "SWIMMING",
    "climbing": "CLIMBING",
    "jumping": "JUMPING",
    "fighting": "AGGRESSIVE_ABNORMAL",
    "attacking": "AGGRESSIVE_ABNORMAL",
    "biting": "AGGRESSIVE_ABNORMAL",
    "chasing": "CHASING",
    "herding": "SOCIAL",
}


class X3DBehaviorClassifier:
    """Pretrained X3D-S behavior backend for wildlife video clips.

    Stage 1 intentionally uses the published Kinetics-400 X3D-S weights as a
    general video-motion prior. It does not pretend that Kinetics is a
    wildlife-specific behavior dataset. Only recognizable action labels are
    translated into the application's behavior ontology; unrelated Kinetics
    classes become UNKNOWN.
    """

    MODEL_NAME = "x3d_s"
    MODEL_VERSION = "X3D-S-Kinetics400-v1"
    NUM_FRAMES = 13
    SIDE_SIZE = 182
    CROP_SIZE = 182
    SAMPLE_RATE = 6
    MEAN = (0.45, 0.45, 0.45)
    STD = (0.225, 0.225, 0.225)

    def __init__(self, device: str = "cpu", label_cache: str | Path = "models/behavior/x3d") -> None:
        self.device = torch.device(device)
        if self.device.type != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {device!r}, but CUDA is unavailable.")

        self.cache_dir = Path(label_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.labels = self._load_labels()
        self.model = self._load_model()
        self.model.to(self.device).eval()

    def _load_model(self):
        try:
            model = torch.hub.load(X3D_REPO, self.MODEL_NAME, pretrained=True)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load pretrained X3D-S. The first run needs network access "
                "to download the PyTorchVideo model and its dependencies. "
                "Install fvcore/iopath if the hub import reports missing packages."
            ) from exc
        return model

    def _load_labels(self) -> dict[int, str]:
        cache_path = self.cache_dir / "kinetics_classnames.json"
        if not cache_path.exists():
            try:
                urllib.request.urlretrieve(KINETICS_LABELS_URL, cache_path)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to download the Kinetics-400 class mapping required by X3D. "
                    f"Expected cache file: {cache_path}"
                ) from exc

        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Invalid Kinetics-400 class mapping: expected an object.")

        result: dict[int, str] = {}
        for label, index in payload.items():
            try:
                result[int(index)] = str(label).replace('"', "").strip()
            except (TypeError, ValueError):
                continue
        if not result:
            raise ValueError("Kinetics-400 class mapping contained no usable labels.")
        return result

    @staticmethod
    def _to_numpy(frame: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(frame, Image.Image):
            return np.asarray(frame.convert("RGB"))
        if isinstance(frame, np.ndarray):
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError("Each frame must be an RGB HxWx3 array.")
            return frame
        raise TypeError("Frames must be NumPy arrays or PIL Images.")

    def _preprocess(self, frames: Sequence[np.ndarray | Image.Image]) -> torch.Tensor:
        if not frames:
            raise ValueError("No frames supplied to X3D.")

        clip = [self._to_numpy(frame) for frame in frames]
        if len(clip) < self.NUM_FRAMES:
            clip += [clip[-1]] * (self.NUM_FRAMES - len(clip))
        else:
            indices = np.linspace(0, len(clip) - 1, self.NUM_FRAMES).round().astype(int)
            clip = [clip[i] for i in indices]

        processed = []
        for frame in clip:
            tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
            tensor = TF.resize(
                tensor,
                self.SIDE_SIZE,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.center_crop(tensor, [self.CROP_SIZE, self.CROP_SIZE])
            tensor = TF.normalize(tensor, self.MEAN, self.STD)
            processed.append(tensor)

        # X3D expects B,C,T,H,W.
        return torch.stack(processed, dim=1)

    @torch.inference_mode()
    def predict(self, frames: Sequence[np.ndarray | Image.Image]) -> dict:
        inputs = self._preprocess(frames).unsqueeze(0).to(self.device)
        logits = self.model(inputs)
        probabilities = torch.softmax(logits, dim=-1)[0]
        top_k = min(5, probabilities.numel())
        values, indices = torch.topk(probabilities, k=top_k)

        candidates = []
        for score, index in zip(values.tolist(), indices.tolist()):
            label = self.labels.get(int(index), f"KineticsClass_{index}")
            candidates.append({
                "label": label,
                "confidence": float(score),
            })

        source_label = candidates[0]["label"] if candidates else "UNKNOWN"
        behaviour = _BEHAVIOUR_ALIASES.get(source_label.strip().lower(), "UNKNOWN")

        return {
            "behaviour": behaviour,
            "confidence": float(candidates[0]["confidence"]) if candidates else 0.0,
            "source_label": source_label,
            "top_predictions": candidates,
            "model_version": self.MODEL_VERSION,
            "frames": self.NUM_FRAMES,
        }

    def predict_paths(self, paths: Iterable[str | Path]) -> dict:
        frames = [np.asarray(Image.open(Path(p)).convert("RGB")) for p in paths]
        return self.predict(frames)
