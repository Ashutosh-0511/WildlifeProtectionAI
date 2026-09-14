from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.8-flash"

GEMINI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "species": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["name", "confidence", "evidence"],
            },
        },
        "primary_species": {"type": "string"},
        "behaviors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "behavior": {"type": "string"},
                    "confidence": {"type": "number"},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "behavior",
                    "confidence",
                    "start_seconds",
                    "end_seconds",
                    "evidence",
                ],
            },
        },
        "primary_behavior": {"type": "string"},
        "behavior_confidence": {"type": "number"},
        "human_present": {"type": "boolean"},
        "risk_score": {"type": "integer"},
        "risk_level": {"type": "string"},
        "risk_reasoning": {"type": "string"},
        "action_recommendation": {"type": "string"},
        "uncertainty": {"type": "string"},
    },
    "required": [
        "status",
        "species",
        "primary_species",
        "behaviors",
        "primary_behavior",
        "behavior_confidence",
        "human_present",
        "risk_score",
        "risk_level",
        "risk_reasoning",
        "action_recommendation",
        "uncertainty",
    ],
}

PROMPT = """
You are the wildlife behaviour analyst in a conservation surveillance system.
Analyze the supplied wildlife video directly. Do not invent observations that are
not visible in the video.

Your tasks:
1. Identify the wild animal species visible in the video. Prefer a specific species
   when the visual evidence supports it; otherwise use UNKNOWN.
2. Identify the active behaviour(s) actually demonstrated over time. Use neutral,
   biologically meaningful behaviour names such as standing, walking, running,
   resting, lying, feeding/foraging, drinking, grooming, stalking, hunting,
   chasing, fighting, mating, nursing, social interaction, alert/vigilant,
   territorial display, carrying prey, or other clearly observed behaviour.
3. Use temporal evidence. A behaviour should be supported by multiple moments in
   the clip when possible. Include approximate start/end timestamps in seconds.
4. Explicitly distinguish humans from wildlife. Do not label a wildlife subject as
   human merely because a person appears elsewhere in the scene.
5. Assess immediate human-safety risk from the animal's observed behaviour and
   proximity/context. This is decision support, not a guarantee of safety.
6. Be conservative: lower confidence or use UNKNOWN when video quality or duration
   does not support a reliable conclusion.

Return ONLY the requested JSON object. Confidence values must be in [0,1].
risk_score must be an integer from 1 to 10. risk_level must be LOW, MEDIUM, HIGH,
or CRITICAL.
""".strip()


def _client(api_key: str | None = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it as an environment variable before running Gemini analysis."
        )
    return genai.Client(api_key=key)


def analyze_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Upload a local video to Gemini and return a validated structured analysis."""
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    client = _client(api_key)
    model_name = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL

    uploaded = client.files.upload(file=str(path))
    try:
        deadline = time.monotonic() + timeout_seconds
        while True:
            state = getattr(uploaded.state, "name", str(uploaded.state))
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Gemini video processing failed after upload")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for Gemini to finish processing the uploaded video")
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        response = client.models.generate_content(
            model=model_name,
            contents=[uploaded, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GEMINI_SCHEMA,
                temperature=0.2,
            ),
        )

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response")

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Gemini response was not a JSON object")

        # Defensive normalization for dashboard stability.
        data["model"] = model_name
        data["source"] = "gemini_video"
        data["video_file"] = path.name
        data["generated_at_epoch"] = time.time()
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            # Cleanup failure should never erase a valid model response.
            pass
