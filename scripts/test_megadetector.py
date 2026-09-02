from __future__ import annotations
import argparse
import json
from pathlib import Path
from ml.detection.megadetector import DetectorConfig, MegaDetectorAdapter

p=argparse.ArgumentParser(description="Run one real MegaDetector V6 inference")
p.add_argument("image", help="Path to a JPG/PNG image")
p.add_argument("--version", default="MDV6-yolov9-c")
p.add_argument("--device", default="auto")
a=p.parse_args()
if not Path(a.image).exists(): raise SystemExit(f"Image not found: {a.image}")
model=MegaDetectorAdapter(DetectorConfig(device=a.device, version=a.version))
result=model.predict(a.image)
print(json.dumps(result, indent=2, default=str))
print("MEGADETECTOR_INFERENCE_OK")
