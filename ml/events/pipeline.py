from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from ml.risk_engine.engine import RiskEngine


@dataclass
class SurveillanceEvent:
    event_id: str
    camera_id: str
    zone_id: str
    species: str
    behaviour: str
    human_present: bool
    distance_m: float | None
    risk_score: float
    risk_level: str
    factors: list[dict[str, Any]]
    evidence_uri: str | None
    created_at: str
    detector_version: str
    tracker_version: str
    behaviour_model_version: str
    risk_model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventOrchestrator:
    """Combines model outputs into an auditable risk event.

    This layer deliberately does not make enforcement decisions.
    """

    def __init__(self, risk_engine: RiskEngine | None = None):
        self.risk_engine = risk_engine or RiskEngine()
        self._recent: dict[str, datetime] = {}

    @staticmethod
    def _event_key(camera_id: str, track_id: str | None, species: str, behaviour: str) -> str:
        raw = f"{camera_id}|{track_id or 'none'}|{species.lower()}|{behaviour.upper()}"
        return sha256(raw.encode("utf-8")).hexdigest()[:20]

    def build_event(
        self,
        *,
        camera_id: str,
        zone_id: str,
        track_id: str | None,
        species: str,
        behaviour: str,
        human_present: bool,
        distance_m: float | None,
        detector_confidence: float,
        behaviour_confidence: float,
        persistence_s: float,
        evidence_uri: str | None = None,
        detector_version: str = "unknown",
        tracker_version: str = "ByteTrack",
        behaviour_model_version: str = "unknown",
        risk_model_version: str = "risk-v1",
        zone_type: str = "forest_core",
        observed_at: datetime | None = None,
    ) -> SurveillanceEvent:
        observed_at = observed_at or datetime.now(timezone.utc)
        risk = self.risk_engine.score(
            species=species,
            behaviour=behaviour,
            human_present=human_present,
            distance_m=distance_m,
            zone_type=zone_type,
            persistence_s=persistence_s,
            detector_confidence=detector_confidence,
            behaviour_confidence=behaviour_confidence,
            observed_at=observed_at,
        )
        key = self._event_key(camera_id, track_id, species, behaviour)
        event_id = f"evt_{key}_{observed_at.strftime('%Y%m%dT%H%M%S%fZ')}"
        return SurveillanceEvent(
            event_id=event_id,
            camera_id=camera_id,
            zone_id=zone_id,
            species=species,
            behaviour=behaviour,
            human_present=human_present,
            distance_m=distance_m,
            risk_score=risk.risk_score,
            risk_level=risk.risk_level,
            factors=risk.factors,
            evidence_uri=evidence_uri,
            created_at=observed_at.isoformat(),
            detector_version=detector_version,
            tracker_version=tracker_version,
            behaviour_model_version=behaviour_model_version,
            risk_model_version=risk_model_version,
        )

    def should_alert(self, event: SurveillanceEvent, dedup_seconds: int = 60) -> bool:
        if event.risk_level not in {"HIGH", "CRITICAL", "UNKNOWN"}:
            return False
        now = datetime.fromisoformat(event.created_at)
        key = sha256(
            f"{event.camera_id}|{event.species}|{event.behaviour}|{event.zone_id}".encode()
        ).hexdigest()[:20]
        previous = self._recent.get(key)
        self._recent[key] = now
        return previous is None or (now - previous).total_seconds() >= dedup_seconds

    @staticmethod
    def serialize(event: SurveillanceEvent) -> dict[str, Any]:
        return event.to_dict()
