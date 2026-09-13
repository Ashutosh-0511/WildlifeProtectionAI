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
DETECTOR_INPUT_SIZE = 1280
DEFAULT_CONFIDENCE = 0.10


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


def _upscale_for_detection(frame: np.ndarray, target_size: int = DETECTOR_INPUT_SIZE) -> tuple[np.ndarray, float, float]:
    """Upscale small frames for detection and return x/y scale factors."""
    height, width = frame.shape[:2]
    if max(width, height) >= target_size:
        return frame, 1.0, 1.0
    scale_x = target_size / max(width, 1)
    scale_y = target_size / max(height, 1)
    resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return resized, scale_x, scale_y


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one xyxy box and an array of xyxy boxes."""
    if boxes.size == 0:
        return np.empty((0,), dtype=float)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.divide(inter, np.maximum(union, 1e-9))


def _tracker_ids_for_raw_detections(raw: sv.Detections, tracked: sv.Detections) -> dict[int, int]:
    """Map tracker IDs back to raw detector indices by geometry, not array position.

    Supervision may filter detections before returning tracked results. Indexing
    tracked.tracker_id with the raw-detection index can therefore assign a
    person's track ID to an animal box (or vice versa). Match the tracked boxes
    back to the original detector boxes using class-aware IoU.
    """
    mapping: dict[int, int] = {}
    if len(raw) == 0 or len(tracked) == 0 or tracked.tracker_id is None:
        return mapping

    raw_boxes = np.asarray(raw.xyxy, dtype=float)
    tracked_boxes = np.asarray(tracked.xyxy, dtype=float)
    raw_classes = np.asarray(raw.class_id) if raw.class_id is not None else None
    tracked_classes = np.asarray(tracked.class_id) if tracked.class_id is not None else None
    used_raw: set[int] = set()

    candidates: list[tuple[float, int, int, int]] = []
    for tracked_idx, tracked_box in enumerate(tracked_boxes):
        tracker_value = tracked.tracker_id[tracked_idx]
        if tracker_value is None:
            continue
        ious = _box_iou_one_to_many(tracked_box, raw_boxes)
        for raw_idx, iou in enumerate(ious):
            if raw_idx in used_raw:
                continue
            if raw_classes is not None and tracked_classes is not None and raw_classes[raw_idx] != tracked_classes[tracked_idx]:
                continue
            candidates.append((float(iou), int(tracked_idx), int(raw_idx), int(tracker_value)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for iou, _tracked_idx, raw_idx, tracker_value in candidates:
        if iou < 0.50 or raw_idx in used_raw:
            continue
        mapping[raw_idx] = tracker_value
        used_raw.add(raw_idx)
    return mapping


def run_video(input_path: Path, output_dir: Path, sample_every: int = 3, confidence: float = DEFAULT_CONFIDENCE) -> dict[str, Any]:
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
    tracker = ByteTrackAdapter(track_thresh=confidence)
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
                detection_frame, scale_x, scale_y = _upscale_for_detection(frame)
                rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                result = detector.model.single_image_detection(rgb, det_conf_thres=confidence)
                detections = _detections_from_result(result)

                # Convert detector boxes back to source-video coordinates before tracking/storage.
                if len(detections) > 0 and (scale_x != 1.0 or scale_y != 1.0):
                    detections.xyxy[:, [0, 2]] /= scale_x
                    detections.xyxy[:, [1, 3]] /= scale_y

                tracked = tracker.update(detections)
                tracker_map = _tracker_ids_for_raw_detections(detections, tracked)

                raw_boxes = detections.xyxy
                raw_class_ids = detections.class_id
                raw_confidences = detections.confidence

                # Build rows from RAW detector output. Tracker IDs are attached
                # to raw detections only after an explicit IoU/class match.
                # Unmatched detections receive a deterministic fallback ID.
                for i, box in enumerate(raw_boxes):
                    cls_id = int(raw_class_ids[i]) if raw_class_ids is not None else -1
                    score = float(raw_confidences[i]) if raw_confidences is not None else 0.0
                    tracker_value = tracker_map.get(i)
                    tracked_flag = tracker_value is not None
                    track_id = str(tracker_value) if tracked_flag else f"det_{frame_index}_{i}"

                    x1, y1, x2, y2 = [float(v) for v in box]
                    row = {
                        "frame_index": frame_index,
                        "timestamp_s": frame_index / fps,
                        "class_id": cls_id,
                        "class_name": DEFAULT_LABELS.get(cls_id, f"class_{cls_id}"),
                        "confidence": score,
                        "bbox": [x1, y1, x2, y2],
                        "track_id": track_id,
                        "tracked": tracked_flag,
                    }
                    detection_rows.append(row)
                    track_rows.append(row.copy())

                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    label = f"#{track_id} {row['class_name']} {score:.2f}"
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
        "detector_input_size": DETECTOR_INPUT_SIZE,
        "detector_confidence": confidence,
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
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")
    summary = run_video(args.input, args.output_dir, args.sample_every, args.confidence)
    print(json.dumps(summary, indent=2))
    print("VIDEO_PIPELINE_OK")


if __name__ == "__main__":
    main()
