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

# Conservative temporal recovery. Behavior clips must not mix two animals.
MAX_FALLBACK_STITCH_GAP = 6
FALLBACK_MIN_SCORE = 0.34
FALLBACK_DUPLICATE_IOU = 0.50
FALLBACK_DUPLICATE_CENTER = 0.20
FALLBACK_MIN_AREA_RATIO = 0.33
FALLBACK_MAX_AREA_RATIO = 3.00
FALLBACK_MIN_ASPECT_RATIO = 0.45
FALLBACK_MAX_ASPECT_RATIO = 2.20
FALLBACK_CENTER_THRESHOLD = 0.45
FALLBACK_IOU_THRESHOLD = 0.12


def _detections_from_result(result: Any) -> sv.Detections:
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


def _upscale_for_detection(
    frame: np.ndarray, target_size: int = DETECTOR_INPUT_SIZE
) -> tuple[np.ndarray, float, float]:
    height, width = frame.shape[:2]
    if max(width, height) >= target_size:
        return frame, 1.0, 1.0
    scale_x = target_size / max(width, 1)
    scale_y = target_size / max(height, 1)
    resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return resized, scale_x, scale_y


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
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


def _box_dimensions(box: list[float]) -> tuple[float, float]:
    return max(1.0, float(box[2]) - float(box[0])), max(1.0, float(box[3]) - float(box[1]))


def _center_distance_normalized(box_a: list[float], box_b: list[float]) -> float:
    ax, ay = _box_center(box_a)
    bx, by = _box_center(box_b)
    aw, ah = _box_dimensions(box_a)
    bw, bh = _box_dimensions(box_b)
    diagonal = max((aw * aw + ah * ah) ** 0.5, (bw * bw + bh * bh) ** 0.5, 1.0)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / diagonal


def _area_ratio(box_a: list[float], box_b: list[float]) -> float:
    a = _box_area(box_a)
    b = _box_area(box_b)
    if a <= 1.0 or b <= 1.0:
        return float("inf")
    return b / a


def _aspect_ratio_ratio(box_a: list[float], box_b: list[float]) -> float:
    aw, ah = _box_dimensions(box_a)
    bw, bh = _box_dimensions(box_b)
    return (bw / bh) / max(aw / ah, 1e-9)


def _is_fallback_id(track_id: Any) -> bool:
    return str(track_id or "").startswith("det_")


def _association_score(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
    """Score a fallback-to-track association using temporal geometry."""
    if previous.get("class_name") != current.get("class_name"):
        return None
    if previous.get("class_name") != "animal":
        return None

    gap = int(current["frame_index"]) - int(previous["frame_index"])
    if gap <= 0 or gap > MAX_FALLBACK_STITCH_GAP:
        return None

    prev_box = list(previous["bbox"])
    curr_box = list(current["bbox"])
    iou = _box_iou(prev_box, curr_box)
    center_distance = _center_distance_normalized(prev_box, curr_box)
    area_ratio = _area_ratio(prev_box, curr_box)
    aspect_ratio = _aspect_ratio_ratio(prev_box, curr_box)

    if not (FALLBACK_MIN_AREA_RATIO <= area_ratio <= FALLBACK_MAX_AREA_RATIO):
        return None
    if not (FALLBACK_MIN_ASPECT_RATIO <= aspect_ratio <= FALLBACK_MAX_ASPECT_RATIO):
        return None
    if iou < FALLBACK_IOU_THRESHOLD and center_distance > FALLBACK_CENTER_THRESHOLD:
        return None

    iou_component = min(1.0, iou / max(FALLBACK_DUPLICATE_IOU, 1e-6))
    center_component = max(0.0, 1.0 - center_distance / FALLBACK_CENTER_THRESHOLD)
    scale_component = max(
        0.0,
        1.0 - abs(np.log(max(area_ratio, 1e-9))) / np.log(FALLBACK_MAX_AREA_RATIO),
    )
    gap_component = 1.0 - (gap - 1) / max(1, MAX_FALLBACK_STITCH_GAP)
    return float(
        0.45 * iou_component
        + 0.35 * center_component
        + 0.15 * scale_component
        + 0.05 * max(0.0, gap_component)
    )


def _same_frame_duplicate(track_row: dict[str, Any], fallback_row: dict[str, Any]) -> bool:
    """Return True when fallback is effectively the same object as a valid track."""
    if track_row.get("class_name") != fallback_row.get("class_name"):
        return False
    track_box = list(track_row["bbox"])
    fallback_box = list(fallback_row["bbox"])
    iou = _box_iou(track_box, fallback_box)
    center_distance = _center_distance_normalized(track_box, fallback_box)
    area_ratio = _area_ratio(track_box, fallback_box)
    return iou >= FALLBACK_DUPLICATE_IOU or (
        center_distance <= FALLBACK_DUPLICATE_CENTER
        and FALLBACK_MIN_AREA_RATIO <= area_ratio <= FALLBACK_MAX_AREA_RATIO
    )


def _stitch_fallback_tracks(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Recover fallback detections with one-to-one, conflict-safe assignment."""
    if not rows:
        return rows, {
            "fallback_rows": 0,
            "matched_existing": 0,
            "new_recovered": 0,
            "duplicate_rejected": 0,
            "ambiguous_rejected": 0,
        }

    frame_groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frame_groups.setdefault(int(row["frame_index"]), []).append(row)

    active: dict[str, dict[str, Any]] = {}
    recovered_counter = 0
    stats = {
        "fallback_rows": 0,
        "matched_existing": 0,
        "new_recovered": 0,
        "duplicate_rejected": 0,
        "ambiguous_rejected": 0,
    }
    kept_rows: list[dict[str, Any]] = []

    for frame_index in sorted(frame_groups):
        current_rows = frame_groups[frame_index]
        stable_rows = [
            row
            for row in current_rows
            if not _is_fallback_id(row.get("track_id")) and row.get("class_name") == "animal"
        ]
        fallback_rows = [
            row
            for row in current_rows
            if _is_fallback_id(row.get("track_id")) and row.get("class_name") == "animal"
        ]
        stats["fallback_rows"] += len(fallback_rows)

        duplicate_indices: set[int] = set()
        for idx, row in enumerate(fallback_rows):
            for stable in stable_rows:
                if _same_frame_duplicate(stable, row):
                    row["track_id_original"] = str(row["track_id"])
                    row["track_recovered"] = False
                    row["tracking_ignored"] = True
                    row["tracking_ignore_reason"] = "duplicate_of_valid_bytetrack"
                    row["duplicate_of_track_id"] = str(stable["track_id"])
                    duplicate_indices.add(idx)
                    stats["duplicate_rejected"] += 1
                    break

        # Existing ByteTrack observations are authoritative.
        for stable in stable_rows:
            tid = str(stable["track_id"])
            active[tid] = stable
            stable["track_recovered"] = False
            stable["tracking_ignored"] = False
            kept_rows.append(stable)

        # Build all candidate edges first, then solve greedily by descending score.
        candidates: list[tuple[float, int, str]] = []
        for row_idx, row in enumerate(fallback_rows):
            if row_idx in duplicate_indices:
                continue
            for tid, previous in list(active.items()):
                score = _association_score(previous, row)
                if score is None or score < FALLBACK_MIN_SCORE:
                    continue
                # Prefer a real ByteTrack identity over a synthetic one when
                # geometry is otherwise comparable.
                if not tid.startswith("recovered_"):
                    score += 0.06
                candidates.append((score, row_idx, tid))
        candidates.sort(key=lambda item: item[0], reverse=True)

        assigned_rows: set[int] = set()
        assigned_tracks: set[str] = set()
        assignment: dict[int, str] = {}
        assignment_score: dict[int, float] = {}
        candidate_scores_by_row: dict[int, list[tuple[float, str]]] = {}
        for score, row_idx, tid in candidates:
            candidate_scores_by_row.setdefault(row_idx, []).append((score, tid))
            if row_idx in assigned_rows or tid in assigned_tracks:
                continue
            assigned_rows.add(row_idx)
            assigned_tracks.add(tid)
            assignment[row_idx] = tid
            assignment_score[row_idx] = score

        # Reject genuinely ambiguous matches instead of forcing identity.
        for row_idx, tid in list(assignment.items()):
            alternatives = [
                (score, candidate_tid)
                for score, candidate_tid in candidate_scores_by_row.get(row_idx, [])
                if candidate_tid != tid
            ]
            if alternatives:
                best = assignment_score[row_idx]
                alt_score, _ = max(alternatives, key=lambda item: item[0])
                if abs(best - alt_score) < 0.035:
                    assignment.pop(row_idx, None)
                    assignment_score.pop(row_idx, None)
                    stats["ambiguous_rejected"] += 1

        for row_idx, row in enumerate(fallback_rows):
            if row_idx in duplicate_indices:
                continue

            original_id = str(row["track_id"])
            row["track_id_original"] = original_id
            row["tracking_ignored"] = False

            tid = assignment.get(row_idx)
            if tid is None:
                recovered_counter += 1
                tid = f"recovered_{recovered_counter}"
                row["track_recovered"] = True
                stats["new_recovered"] += 1
            else:
                row["track_recovered"] = tid.startswith("recovered_")
                stats["matched_existing"] += 1

            row["track_id"] = tid
            active[tid] = row
            kept_rows.append(row)

        # Expire identities after a real temporal gap.
        for tid in [
            tid
            for tid, previous in active.items()
            if frame_index - int(previous["frame_index"]) > MAX_FALLBACK_STITCH_GAP
        ]:
            active.pop(tid, None)

    return kept_rows, stats


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
            if (
                raw_classes is not None
                and tracked_classes is not None
                and raw_classes[raw_idx] != tracked_classes[tracked_idx]
            ):
                continue
            candidates.append((float(iou), int(tracked_idx), int(raw_idx), int(tracker_value)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for iou, _tracked_idx, raw_idx, tracker_value in candidates:
        if iou < 0.50 or raw_idx in used_raw:
            continue
        mapping[raw_idx] = tracker_value
        used_raw.add(raw_idx)
    return mapping


def run_video(
    input_path: Path,
    output_dir: Path,
    sample_every: int = 3,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
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
    raw_track_rows: list[dict[str, Any]] = []
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
                    detection_rows.append(row.copy())
                    raw_track_rows.append(row)

                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    label = f"#{track_id} {row['class_name']} {score:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (p1[0], max(20, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    track_rows, recovery_stats = _stitch_fallback_tracks(raw_track_rows)

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
        "unique_track_ids": sorted({r["track_id"] for r in track_rows if r.get("track_id") is not None}),
        "detector": "MegaDetectorV6",
        "detector_version": "MDV6-yolov9-c",
        "detector_input_size": DETECTOR_INPUT_SIZE,
        "detector_confidence": confidence,
        "tracker": "ByteTrack",
        "temporal_recovery": "spatial_stitch_v2",
        "temporal_recovery_max_gap": MAX_FALLBACK_STITCH_GAP,
        "temporal_recovery_policy": {
            "one_to_one_assignment": True,
            "duplicate_of_valid_bytetrack_rejected": True,
            "ambiguous_matches_rejected": True,
            "valid_bytetrack_ids_preserved": True,
        },
        "temporal_recovery_stats": recovery_stats,
        "device": "cpu",
    }

    detections_path.write_text(
        json.dumps({"summary": summary, "detections": detection_rows}, indent=2),
        encoding="utf-8",
    )
    tracks_path.write_text(
        json.dumps({"summary": summary, "tracks": track_rows}, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MegaDetector + ByteTrack on a video")
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/video"))
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    args = parser.parse_args()

    summary = run_video(
        input_path=args.input,
        output_dir=args.output_dir,
        sample_every=max(1, args.sample_every),
        confidence=args.confidence,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
