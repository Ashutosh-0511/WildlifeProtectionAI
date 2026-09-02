from __future__ import annotations


class BehaviorMapper:
    """Stable application ontology for behaviour predictions."""

    MAP = {
        "Standing": "STATIONARY",
        "Lying": "RESTING",
        "Foraging/Grazing": "FEEDING",
        "Drinking": "DRINKING",
        "Ruminating": "RESTING_FEEDING",
        "Grooming": "GROOMING",
        "Other": "UNKNOWN",
    }

    @classmethod
    def map(cls, behaviour: str) -> str:
        return cls.MAP.get(behaviour, "UNKNOWN")

    @classmethod
    def enrich(cls, prediction: dict) -> dict:
        result = dict(prediction)
        result["behavior_class"] = cls.map(prediction.get("behaviour", "Other"))
        return result
