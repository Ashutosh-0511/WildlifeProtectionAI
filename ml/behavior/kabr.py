from __future__ import annotations

import csv
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2


KABR_CLASSES = (
    "Walk",
    "Graze",
    "Browse",
    "Head Up",
    "Auto-Groom",
    "Trot",
    "Run",
    "Occluded",
)


class KABRBehaviorInference:
    """Adapter for the official Imageomics kabr-tools X3D-KABR model.

    The adapter intentionally runs kabr-tools as a subprocess so the main
    project environment can remain on Python 3.13/CPU while kabr-tools may use
    its supported Python 3.10/3.11 environment.
    """

    def __init__(
        self,
        python_executable: str | Path,
        model_hub: str = "imageomics/x3d-kabr-kinetics",
        checkpoint: str = "x3d-l-kabr-kinetics-lr5e-2-ep120-best-epoch.zip",
    ):
        self.python_executable = str(python_executable)
        self.model_hub = model_hub
        self.checkpoint = checkpoint

    def check_installation(self) -> None:
        probe = subprocess.run(
            [self.python_executable, "-c", "import kabr_tools; print('KABR_TOOLS_OK')"],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "kabr-tools is unavailable in the configured Python environment. "
                f"Python: {self.python_executable}\n"
                f"Error: {probe.stderr.strip() or probe.stdout.strip()}"
            )

    @staticmethod
    def _write_miniscene(frames: list[Path], root: Path, track_id: str) -> tuple[Path, Path]:
        track_root = root / "track"
        track_root.mkdir(parents=True, exist_ok=True)
        video_path = track_root / f"{track_id}.mp4"
        metadata = track_root / "metadata"
        metadata.mkdir(parents=True, exist_ok=True)
        xml_path = metadata / "track_tracks.xml"

        first = cv2.imread(str(frames[0]))
        if first is None:
            raise RuntimeError(f"Unable to read behaviour crop: {frames[0]}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create KABR mini-scene: {video_path}")

        try:
            valid = 0
            for frame_path in frames:
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue
                image = cv2.resize(image, (width, height))
                writer.write(image)
                valid += 1
        finally:
            writer.release()

        if valid < 2:
            raise RuntimeError(f"Need at least 2 valid frames for {track_id}")

        root_xml = ET.Element("annotations")
        track = ET.SubElement(root_xml, "track", {"id": str(track_id), "label": "animal"})
        for i in range(valid):
            ET.SubElement(
                track,
                "box",
                {"frame": str(i), "xtl": "0", "ytl": "0",
                 "xbr": str(width), "ybr": str(height)},
            )
        ET.ElementTree(root_xml).write(xml_path, encoding="utf-8", xml_declaration=True)
        return track_root, video_path

    def predict(self, frame_paths: list[str | Path], work_dir: str | Path, track_id: str) -> dict[str, Any]:
        frames = [Path(p) for p in frame_paths]
        if len(frames) < 2:
            return {
                "behaviour": "UNKNOWN",
                "confidence": 0.0,
                "frames": len(frames),
                "model_version": "X3D-KABR-Kinetics",
                "reason": "insufficient_frames",
            }

        work = Path(work_dir)
        mini_root, _ = self._write_miniscene(frames, work, track_id)
        output_csv = work / f"{track_id}_behavior.csv"

        command = [
            self.python_executable,
            "-m",
            "kabr_tools.miniscene2behavior",
            "--hub",
            self.model_hub,
            "--checkpoint",
            self.checkpoint,
            "--miniscene",
            str(mini_root),
            "--video",
            "track",
            "--output",
            str(output_csv),
        ]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "KABR behavior inference failed.\n"
                + (proc.stderr.strip() or proc.stdout.strip())
            )

        if not output_csv.exists():
            raise RuntimeError(f"KABR did not produce {output_csv}")

        labels: list[int] = []
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            sample = handle.read(2048)
            handle.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=" ,\t")
            reader = csv.DictReader(handle, dialect=dialect)
            for row in reader:
                value = row.get("label")
                if value is not None:
                    try:
                        labels.append(int(value))
                    except ValueError:
                        pass

        if not labels:
            raise RuntimeError(f"KABR output contained no labels: {output_csv}")

        # The official KABR classifier emits one of eight class indices.
        counts = {i: labels.count(i) for i in range(len(KABR_CLASSES))}
        best = max(counts, key=counts.get)
        confidence = counts[best] / len(labels)
        return {
            "behaviour": KABR_CLASSES[best],
            "confidence": float(confidence),
            "frames": len(labels),
            "model_version": "X3D-KABR-Kinetics",
            "class_index": int(best),
            "label_counts": counts,
        }
