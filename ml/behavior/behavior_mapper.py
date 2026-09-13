from __future__ import annotations


class BehaviorMapper:
    """Stable application ontology for behaviour predictions."""

    MAP = {
        # Legacy VideoMAE labels retained for compatibility.
        "Standing": "STATIONARY",
        "Lying": "RESTING",
        "Foraging/Grazing": "FEEDING",
        "Drinking": "DRINKING",
        "Ruminating": "RESTING_FEEDING",
        "Grooming": "GROOMING",
        "Other": "UNKNOWN",
        # X3D already emits canonical classes; keep them stable through the
        # same mapper so the downstream risk engine and API remain unchanged.
        "STATIONARY": "STATIONARY",
        "RESTING": "RESTING",
        "FEEDING": "FEEDING",
        "DRINKING": "DRINKING",
        "GROOMING": "GROOMING",
        "RUNNING": "RUNNING",
        "NORMAL_MOVEMENT": "NORMAL_MOVEMENT",
        "CHASING": "CHASING",
        "AGGRESSIVE_ABNORMAL": "AGGRESSIVE_ABNORMAL",
        "SWIMMING": "SWIMMING",
        "CLIMBING": "CLIMBING",
        "JUMPING": "JUMPING",
        "SOCIAL": "SOCIAL",
        "UNKNOWN": "UNKNOWN",
    }

    @classmethod
    def map(cls, behaviour: str) -> str:
        return cls.MAP.get(behaviour, "UNKNOWN")

    @classmethod
    def enrich(cls, prediction: dict) -> dict:
        result = dict(prediction)
        result["behavior_class"] = cls.map(prediction.get("behaviour", "Other"))
        return result
