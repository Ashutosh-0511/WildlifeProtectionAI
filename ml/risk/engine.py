from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RiskInput:
    species: str
    behaviour: str
    human_present: bool
    distance_m: float | None
    zone_risk: float = 0.0
    time_risk: float = 0.0
    persistence_s: float = 0.0
    confidence: float = 1.0
    detector_confidence: float = 1.0
    behaviour_confidence: float = 1.0


SPECIES_WEIGHT = {
    "elephant": 0.20,
    "tiger": 0.25,
    "leopard": 0.20,
    "bear": 0.15,
    "unknown": 0.05,
}
BEHAVIOUR_WEIGHT = {
    "RESTING": 0.0,
    "NORMAL_MOVEMENT": 0.05,
    "RUNNING": 0.20,
    "CHASING": 0.40,
    "AGGRESSIVE_ABNORMAL": 0.50,
    "UNKNOWN": 0.0,
}


def score_risk(x: RiskInput) -> dict[str, Any]:
    factors: list[dict[str, float | str]] = []
    score = 0.0

    species = x.species.strip().lower()
    sw = SPECIES_WEIGHT.get(species, SPECIES_WEIGHT["unknown"])
    score += sw
    factors.append({"name": "species", "value": x.species, "contribution": sw})

    behaviour = x.behaviour.strip().upper()
    bw = BEHAVIOUR_WEIGHT.get(behaviour, 0.0)
    score += bw
    factors.append({"name": "behaviour", "value": x.behaviour, "contribution": bw})

    hp = 0.25 if x.human_present else 0.0
    score += hp
    factors.append({"name": "human_presence", "value": float(x.human_present), "contribution": hp})

    proximity = 0.0 if x.distance_m is None else max(0.0, min(0.25, 0.25 * (1.0 - x.distance_m / 100.0)))
    score += proximity
    factors.append({"name": "proximity", "value": x.distance_m if x.distance_m is not None else -1.0, "contribution": proximity})

    persistence = min(0.10, max(0.0, x.persistence_s / 60.0 * 0.10))
    score += persistence
    factors.append({"name": "persistence", "value": x.persistence_s, "contribution": persistence})

    zone = max(0.0, min(0.10, x.zone_risk))
    time = max(0.0, min(0.10, x.time_risk))
    score += zone
    score += time
    factors.append({"name": "zone", "value": x.zone_risk, "contribution": zone})
    factors.append({"name": "time", "value": x.time_risk, "contribution": time})

    evidence_conf = max(0.0, min(1.0, min(
        x.confidence, x.detector_confidence, x.behaviour_confidence
    )))
    uncertainty = max(0.0, min(0.20, 0.20 * (1.0 - evidence_conf)))
    score -= uncertainty
    factors.append({"name": "uncertainty_penalty", "value": evidence_conf, "contribution": -uncertainty})

    score = max(0.0, min(1.0, score))
    if behaviour == "UNKNOWN" or evidence_conf < 0.35:
        level = "MEDIUM" if x.human_present else "UNKNOWN"
    elif score >= 0.75:
        level = "CRITICAL"
    elif score >= 0.50:
        level = "HIGH"
    elif score >= 0.25:
        level = "MEDIUM"
    else:
        level = "LOW"
    return {"risk_score": round(score, 4), "risk_level": level, "factors": factors}
