from __future__ import annotations

import time
import urllib.error
from dataclasses import dataclass
from typing import Any


@dataclass
class DetectorConfig:
    confidence: float = 0.25
    device: str = "auto"
    version: str = "MDV6-yolov9-c"
    load_retries: int = 4
    retry_delay_s: float = 3.0


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

        attempts = max(1, int(self.config.load_retries))
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                self.model = pw_detection.MegaDetectorV6(
                    device=device,
                    pretrained=True,
                    version=self.config.version,
                )
                return
            except urllib.error.HTTPError as exc:
                last_error = exc
                if attempt == attempts:
                    raise RuntimeError(
                        f"MegaDetector weights could not be downloaded after {attempts} attempts "
                        f"(HTTP {exc.code}). The model provider returned a transient download error."
                    ) from exc
                time.sleep(self.config.retry_delay_s * attempt)
            except Exception:
                raise

        if last_error is not None:
            raise last_error

    def predict(self, image_path: str) -> Any:
        if self.model is None:
            self.load()
        return self.model.single_image_detection(image_path)
