from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class DetectorConfig:
    confidence: float = 0.25
    device: str = "auto"
    version: str = "MDV6-yolov9-c"

class MegaDetectorAdapter:
    """MegaDetector V6 adapter using the official PyTorch-Wildlife API."""
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.model: Any = None

    def load(self) -> None:
        try:
            from PytorchWildlife.models import detection as pw_detection
        except ImportError as exc:
            raise RuntimeError("Install PytorchWildlife before loading MegaDetector") from exc
        device = self.config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pw_detection.MegaDetectorV6(
            device=device, pretrained=True, version=self.config.version
        )

    def predict(self, image_path: str) -> Any:
        if self.model is None:
            self.load()
        return self.model.single_image_detection(image_path)
