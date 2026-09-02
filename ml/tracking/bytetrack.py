from __future__ import annotations

from typing import Any


class ByteTrackAdapter:
    """Detector-independent ByteTrack adapter backed by Supervision."""

    def __init__(self, track_thresh: float = 0.25, track_buffer: int = 30, match_thresh: float = 0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self._tracker: Any = None

    def load(self) -> None:
        try:
            from supervision import ByteTrack
        except ImportError as exc:
            raise RuntimeError("Install supervision to use the ByteTrack adapter") from exc
        self._tracker = ByteTrack(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
        )

    def update(self, detections: Any) -> Any:
        if self._tracker is None:
            self.load()
        return self._tracker.update_with_detections(detections)
