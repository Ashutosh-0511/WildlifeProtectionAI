from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

from ml.detection.megadetector import DetectorConfig, MegaDetectorAdapter
from ml.tracking.bytetrack import ByteTrackAdapter


DEFAULT_LABELS = {0: "animal", 1: "person", 2: "vehicle"}


def _detections_from_result(result: Any) -> sv.Detections:
    """Extract the Supervision Detections object returned by PyTorch-Wildlife."""
    if isinstance(result, dict) and isinstance(result.get("detections"), sv.Detections):
        return result["detections"]
    if isinstance(result, sv.Detections):
        return result
    raise RuntimeError(
        "Unexpected MegaDetector output. Expected a dict containing "
        "a Supervision Detections object under 'detections'."
    )


def _to_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_video(input_path: Path, output_dir: Path, sample_every: int = 3, confidence: float = 0.25) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "annotated.mp4"
    detections_path = output_dir / "detections.json"
    tracks_path = output_dir / "tracks.json"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to create output video: {annotated_path}")

    detector = MegaDetectorAdapter(DetectorConfig(confidence=confidence, device="cpu"))
    detector.load()
    tracker = ByteTrackAdapter()
    tracker.load()

    detection_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    frame_index = 0
    sampled_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % max(1, sample_every) == 0:
                sampled_frames += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.model.single_image_detection(rgb, det_conf_thres=confidence)
                detections = _detections_from_result(result)

                # ByteTrack expects xyxy + confidence + class_id.
                tracked = tracker.update(detections)
                tracker_ids = tracked.tracker_id
                class_ids = tracked.class_id
                confidences = tracked.confidence
                boxes = tracked.xyxy

                for i, box in enumerate(boxes):
                    cls_id = int(class_ids[i]) if class_ids is not None else -1
                    score = float(confidences[i]) if confidences is not None else 0.0
                    track_id = None
                    if tracker_ids is not None and tracker_ids[i] is not None:
                        track_id = str(int(tracker_ids[i]))
                    x1, y1, x2, y2 = [float(v) for v in box]
                    row = {
                        "frame_index": frame_index,
                        "timestamp_s": frame_index / fps,
                        "class_id": cls_id,
                        "class_name": DEFAULT_LABELS.get(cls_id, f"class_{cls_id}"),
                        "confidence": score,
                        "bbox": [x1, y1, x2, y2],
                        "track_id": track_id,
                    }
                    detection_rows.append(row)
                    if track_id is not None:
                        track_rows.append(row.copy())

                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    label = f"#{track_id or '-'} {row['class_name']} {score:.2f}"
                    cv2.putText(frame, label, (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    summary = {
        "input": str(input_path),
        "output": str(annotated_path),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames or frame_index,
        "sample_every": sample_every,
        "sampled_frames": sampled_frames,
        "detection_count": len(detection_rows),
        "track_observation_count": len(track_rows),
        "unique_track_ids": sorted({r["track_id"] for r in track_rows if r["track_id"] is not None}),
        "detector": "MegaDetectorV6",
        "detector_version": "MDV6-yolov9-c",
        "tracker": "ByteTrack",
        "device": "cpu",
    }
    detections_path.write_text(json.dumps({"summary": summary, "detections": detection_rows}, indent=2), encoding="utf-8")
    tracks_path.write_text(json.dumps({"summary": summary, "tracks": track_rows}, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MegaDetector + ByteTrack on a video.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/video_test"))
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")
    summary = run_video(args.input, args.output_dir, args.sample_every, args.confidence)
    print(json.dumps(summary, indent=2))
    print("VIDEO_PIPELINE_OK")


if __name__ == "__main__":
    main()
