from __future__ import annotations

from typing import Iterable

PROJECT_BEHAVIORS = (
    "RESTING",
    "NORMAL_MOVEMENT",
    "RUNNING",
    "CHASING",
    "AGGRESSIVE_ABNORMAL",
    "UNKNOWN",
)

# Conservative mapping: source actions only map when their meaning is sufficiently clear.
# Unknown/unmapped actions remain UNKNOWN rather than being assigned a risky class.
ACTION_TO_BEHAVIOR = {
    "resting": "RESTING",
    "sleeping": "RESTING",
    "standing": "RESTING",
    "walking": "NORMAL_MOVEMENT",
    "moving": "NORMAL_MOVEMENT",
    "running": "RUNNING",
    "chasing": "CHASING",
    "attacking": "AGGRESSIVE_ABNORMAL",
    "fighting": "AGGRESSIVE_ABNORMAL",
}


def map_actions(actions: Iterable[str]) -> str:
    mapped = []
    for action in actions:
        key = str(action).strip().lower().replace("-", "_")
        if key in ACTION_TO_BEHAVIOR:
            mapped.append(ACTION_TO_BEHAVIOR[key])
    if not mapped:
        return "UNKNOWN"
    # Prioritize the most operationally significant mapped class when multiple
    # source labels occur in one clip.
    priority = {
        "AGGRESSIVE_ABNORMAL": 5,
        "CHASING": 4,
        "RUNNING": 3,
        "NORMAL_MOVEMENT": 2,
        "RESTING": 1,
        "UNKNOWN": 0,
    }
    return max(set(mapped), key=lambda x: priority[x])
