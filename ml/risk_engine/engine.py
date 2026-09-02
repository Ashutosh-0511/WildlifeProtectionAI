from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class RiskEvent:
    risk_score: float
    risk_level: str
    factors: list[dict[str, Any]]


class RiskEngine:
    """V1 transparent decision-support scorer; weights require calibration."""

    def __init__(self, species_weights=None, behaviour_weights=None):
        self.species_weights = species_weights or {
            "elephant": 20.0, "tiger": 25.0, "leopard": 20.0, "unknown": 5.0
        }
        self.behaviour_weights = behaviour_weights or {
            "RESTING": 0.0, "NORMAL_MOVEMENT": 5.0, "RUNNING": 15.0,
            "CHASING": 30.0, "AGGRESSIVE_ABNORMAL": 40.0, "UNKNOWN": 5.0
        }

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(100.0, value))

    def score(self, *, species: str, behaviour: str, human_present: bool,
              distance_m: float | None = None, zone_type: str = "forest_core",
              hour: int | None = None, persistence_s: float = 0.0,
              detector_confidence: float = 1.0,
              behaviour_confidence: float = 1.0,
              observed_at: datetime | None = None) -> RiskEvent:
        factors = []
        def add(name: str, value: float):
            factors.append({"name": name, "value": value, "contribution": float(value)})

        add("species", self.species_weights.get(species.strip().lower(), 5.0))
        add("behaviour", self.behaviour_weights.get(behaviour.strip().upper(), 5.0))
        add("human_presence", 20.0 if human_present else 0.0)

        if distance_m is None:
            proximity = 0.0
        elif distance_m <= 5:
            proximity = 25.0
        elif distance_m <= 15:
            proximity = 15.0
        elif distance_m <= 30:
            proximity = 7.0
        else:
            proximity = 0.0
        add("proximity", proximity)

        zone_weights = {"forest_core": 0.0, "corridor": 5.0, "road": 10.0,
                        "railway": 15.0, "village_boundary": 20.0, "settlement": 25.0}
        add("zone", zone_weights.get(zone_type.strip().lower(), 5.0))

        effective_hour = hour if hour is not None else (observed_at.hour if observed_at else None)
        add("time", 5.0 if effective_hour is not None and (effective_hour < 6 or effective_hour >= 18) else 0.0)
        add("persistence", min(15.0, max(0.0, persistence_s) / 10.0))

        evidence_conf = max(0.0, min(1.0, min(detector_confidence, behaviour_confidence)))
        add("uncertainty_penalty", -(1.0 - evidence_conf) * 20.0)

        score = self._clamp(sum(f["contribution"] for f in factors))
        if evidence_conf < 0.35:
            level = "UNKNOWN"
        elif score >= 80:
            level = "CRITICAL"
        elif score >= 60:
            level = "HIGH"
        elif score >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"
        return RiskEvent(score, level, factors)
