from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Detection:
    frame_index: int
    timestamp_s: float
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: str | None = None

@dataclass
class BehaviourResult:
    track_id: str
    behaviour: str
    confidence: float
    window_start_s: float
    window_end_s: float
    model_version: str

@dataclass
class RiskEvent:
    risk_event_id: str
    camera_id: str
    zone_id: str
    species: str
    behaviour: str
    human_present: bool
    distance_m: float | None
    risk_score: float
    risk_level: str
    factors: list[dict]
    evidence_uri: str | None
