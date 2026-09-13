from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
import urllib.error


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

    def _cached_weight(self) -> Path | None:
        """Return a known PyTorch hub cache file for the configured V6 model."""
        import torch

        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        names = {
            "MDV6-yolov9-c": ("MDV6b-yolov9-c.pt", "MDV6-yolov9-c.pt"),
            "MDV6-yolov9-e": ("MDV6-yolov9-e-1280.pt",),
            "MDV6-yolov10-c": ("MDV6-yolov10-c.pt",),
            "MDV6-yolov10-e": ("MDV6-yolov10-e-1280.pt",),
            "MDV6-rtdetr-c": ("MDV6b-rtdetr-c.pt", "MDV6-rtdetr-c.pt"),
        }.get(self.config.version, ())
        for name in names:
            path = cache_dir / name
            if path.exists() and path.stat().st_size > 5_000_000:
                return path
        return None

    def load(self) -> None:
        try:
            from PytorchWildlife.models import detection as pw_detection
        except ImportError as exc:
            raise RuntimeError("Install PytorchWildlife before loading MegaDetector") from exc

        device = self.config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cached = self._cached_weight()
        if cached is not None:
            try:
                self.model = pw_detection.MegaDetectorV6(
                    weights=str(cached),
                    device=device,
                    pretrained=False,
                    version=self.config.version,
                )
                return
            except Exception:
                # A stale/corrupt cache should not permanently block loading.
                pass

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
                        f"MegaDetector weights could not be loaded after {attempts} attempts "
                        f"(HTTP {exc.code}). No usable cached weights were found."
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
