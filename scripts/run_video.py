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
MAX_FALLBACK_STITCH_GAP = 6
FALLBACK_IOU_THRESHOLD = 0.20
FALLBACK_CENTER_THRESHOLD = 0.35
FALLBACK_MIN_AREA_RATIO = 0.40
FALLBACK_MAX_AREA_RATIO = 2.50


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


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    a = np.asarray(box_a, dtype=float)
    b = np.asarray(box_b, dtype=float)
    return float(_box_iou_one_to_many(a, b.reshape(1, 4))[0])


def _box_center(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5)


def _box_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _fallback_match_score(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
    """Return a conservative spatial continuity score for fallback detections.

    Fallback detector IDs are created when ByteTrack does not return an ID for a
    raw detection. We may reuse a recent track ID only when the boxes are
    geometrically consistent. This avoids turning every one-frame detection into
    a separate behavior track while refusing obviously different animals.
    """
    if previous.get("class_name") != current.get("class_name"):
        return None

    gap = int(current["frame_index"]) - int(previous["frame_index"])
    if gap <= 0 or gap > MAX_FALLBACK_STITCH_GAP:
        return None

    prev_box = list(previous["bbox"])
    curr_box = list(current["bbox"])
    iou = _box_iou(prev_box, curr_box)

    px, py = _box_center(prev_box)
    cx, cy = _box_center(curr_box)
    prev_w = max(1.0, float(prev_box[2]) - float(prev_box[0]))
    prev_h = max(1.0, float(prev_box[3]) - float(prev_box[1]))
    curr_w = max(1.0, float(curr_box[2]) - float(curr_box[0]))
    curr_h = max(1.0, float(curr_box[3]) - float(curr_box[1]))
    diagonal = max((prev_w * prev_w + prev_h * prev_h) ** 0.5, (curr_w * curr_w + curr_h * curr_h) ** 0.5, 1.0)
    center_distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / diagonal

    prev_area = _box_area(prev_box)
    curr_area = _box_area(curr_box)
    if prev_area <= 1.0 or curr_area <= 1.0:
        return None
    area_ratio = curr_area / prev_area
    if not (FALLBACK_MIN_AREA_RATIO <= area_ratio <= FALLBACK_MAX_AREA_RATIO):
        return None

    if iou < FALLBACK_IOU_THRESHOLD and center_distance > FALLBACK_CENTER_THRESHOLD:
        return None

    iou_component = iou
    center_component = max(0.0, 1.0 - center_distance / FALLBACK_CENTER_THRESHOLD)
    gap_penalty = 1.0 - (gap - 1) / max(1, MAX_FALLBACK_STITCH_GAP)
    return (0.65 * iou_component + 0.35 * center_component) * max(0.0, gap_penalty)


def _stitch_fallback_tracks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover short temporal tracks from raw fallback detections.

    Only fallback IDs (`det_<frame>_<index>`) are reassigned. Existing ByteTrack
    IDs are never overwritten by this recovery pass. A fallback observation can
    attach to an existing recent track only when class, time gap, IoU/center
    motion, and bounding-box scale are all plausible.
    """
    if not rows:
        return rows

    frame_groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frame_groups.setdefault(int(row["frame_index"]), []).append(row)

    # Track the latest observation for every known track ID. This lets fallback
    # detections reconnect to a genuine ByteTrack ID after a short association
    # failure without changing already-valid tracker assignments.
    active: dict[str, dict[str, Any]] = {}
    fallback_counter = 0

    for frame_index in sorted(frame_groups):
        current_rows = frame_groups[frame_index]
        used_track_ids: set[str] = set()

        # Stable tracker observations refresh active state first.
        for row in current_rows:
            tid = str(row.get("track_id", ""))
            if not tid.startswith("det_") and tid:
                active[tid] = row
                used_track_ids.add(tid)

        fallback_rows = [
            row for row in current_rows
            if str(row.get("track_id", "")).startswith("det_") and row.get("class_name") == "animal"
        ]
        fallback_rows.sort(key=lambda r: float(r.get("confidence", 0.0)), reverse=True)

        for row in fallback_rows:
            best_tid = None
            best_score = -1.0
            for tid, previous in list(active.items()):
                if tid in used_track_ids:
                    continue
                score = _fallback_match_score(previous, row)
                if score is not None and score > best_score:
                    best_tid = tid
                    best_score = score

            if best_tid is None:
                fallback_counter += 1
                best_tid = f"recovered_{fallback_counter}"

            original_id = str(row["track_id"])
            row["track_id_original"] = original_id
            row["track_recovered"] = best_tid.startswith("recovered_") or best_tid != original_id
            row["track_id"] = best_tid
            active[best_tid] = row
            used_track_ids.add(best_tid)

    return rows


def _tracker_ids_for_raw_detections(raw: sv.Detections, tracked: sv.Detections) -> dict[int, int]:
    """Map tracker IDs back to raw detector indices by geometry, not array position."""
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

                if len(detections) > 0 and (scale_x != 1.0 or scale_y != 1.0):
                    detections.xyxy[:, [0, 2]] /= scale_x
                    detections.xyxy[:, [1, 3]] /= scale_y

                tracked = tracker.update(detections)
                tracker_map = _tracker_ids_for_raw_detections(detections, tracked)

                raw_boxes = detections.xyxy
                raw_class_ids = detections.class_id
                raw_confidences = detections.confidence

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

    # Recover only short-lived fallback associations. This operates after the
    # detector/tracker pass and therefore cannot perturb MegaDetector boxes or
    # valid ByteTrack IDs. It gives the behavior clip builder a chance to form
    # temporal tracks from detections that were spatially consistent but missed
    # by ByteTrack on one or two sampled frames.
    track_rows = _stitch_fallback_tracks(track_rows)
    detection_rows = track_rows.copy()

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
        "temporal_recovery": "spatial_stitch_v1",
        "temporal_recovery_max_gap": MAX_FALLBACK_STITCH_GAP,
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
