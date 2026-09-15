from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.behavior.gemini_video import AnalysisUnavailableError, analyze_video as analyze_with_gemini
from ml.behavior.llama_vision import analyze_video as analyze_with_llama


def analyze_video_with_fallbacks(
    video_path: str | Path,
    output_path: str | Path | None = None,
    *,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run the three Gemini models first, then NVIDIA Llama Vision as fallback."""
    try:
        return analyze_with_gemini(
            video_path,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
        )
    except AnalysisUnavailableError:
        pass

    try:
        return analyze_with_llama(
            video_path,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        raise AnalysisUnavailableError("All configured video-analysis models were unavailable") from exc
