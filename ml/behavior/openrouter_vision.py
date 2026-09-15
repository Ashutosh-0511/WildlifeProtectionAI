from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import requests
from dotenv import load_dotenv

from ml.behavior.gemini_video import PROMPT

load_dotenv()

DEFAULT_MODEL = "qwen/qwen-2.5-vl-7b-instruct:free"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_FRAMES = 8


class OpenRouterVisionUnavailableError(RuntimeError):
    """Raised when the OpenRouter vision fallback cannot serve the request."""


def _sample_frames(video_path: Path, max_frames: int = MAX_FRAMES) -> list[tuple[float, str]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OpenRouterVisionUnavailableError("Unable to open video for fallback analysis")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            raise OpenRouterVisionUnavailableError("Video contains no readable frames")

        indices = sorted(
            set(
                round(i * (frame_count - 1) / max(max_frames - 1, 1))
                for i in range(max_frames)
            )
        )
        sampled: list[tuple[float, str]] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            success, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78]
            )
            if not success:
                continue
            timestamp = index / fps if fps > 0 else float(index)
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")
            sampled.append((timestamp, f"data:image/jpeg;base64,{payload}"))

        if not sampled:
            raise OpenRouterVisionUnavailableError("Unable to sample readable frames from video")
        return sampled
    finally:
        capture.release()


def _extract_text(response: requests.Response) -> str:
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterVisionUnavailableError("OpenRouter returned no choices")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return str(content).strip()


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json\n", "", 1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise OpenRouterVisionUnavailableError("OpenRouter returned non-JSON output")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise OpenRouterVisionUnavailableError("OpenRouter returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise OpenRouterVisionUnavailableError("OpenRouter response was not a JSON object")
    return parsed


def _write_result(
    data: dict[str, Any],
    output_path: str | Path | None,
    video_path: Path,
    model_name: str,
) -> dict[str, Any]:
    data["status"] = "success"
    data["model"] = model_name
    data["source"] = "video_analysis"
    data["video_file"] = video_path.name
    data["generated_at_epoch"] = time.time()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def analyze_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    *,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise OpenRouterVisionUnavailableError("OpenRouter API key is not configured")

    endpoint = os.getenv("OPENROUTER_BASE_URL", DEFAULT_ENDPOINT).strip() or DEFAULT_ENDPOINT
    model_name = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    max_frames = max(1, int(os.getenv("OPENROUTER_MAX_FRAMES", str(MAX_FRAMES))))
    frames = _sample_frames(path, max_frames)

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"You are analyzing a wildlife video represented by {len(frames)} uniformly sampled frames. "
                "Use the frame order as temporal evidence and estimate timestamps from the provided labels.\n\n"
                + PROMPT
            ),
        }
    ]
    for timestamp, data_uri in frames:
        content.append({"type": "text", "text": f"Frame timestamp: {timestamp:.2f} seconds"})
        content.append({"type": "image_url", "image_url": {"url": data_uri}})

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 1600,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = _parse_json(_extract_text(response))
    except Exception as exc:
        raise OpenRouterVisionUnavailableError("OpenRouter vision fallback was unavailable") from exc

    if not data:
        raise OpenRouterVisionUnavailableError("OpenRouter returned an empty analysis")
    return _write_result(data, output_path, path, model_name)
