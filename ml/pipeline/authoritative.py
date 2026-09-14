from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

from ml.behavior.gemini_video import AnalysisUnavailableError, analyze_video
from ml.pipeline.integrated import run_integrated


def _normalize_species_name(value: Any) -> str:
    name = str(value or "UNKNOWN").strip()
    return name or "UNKNOWN"


def _normalize_behavior_name(value: Any) -> str:
    name = str(value or "UNKNOWN").strip()
    return name or "UNKNOWN"


def _dashboard_result(
    analysis: dict[str, Any],
    local_result: dict[str, Any] | None,
    video: Path,
    output_dir: Path,
) -> dict[str, Any]:
    primary_species = _normalize_species_name(analysis.get("primary_species"))
    primary_behavior = _normalize_behavior_name(analysis.get("primary_behavior"))

    try:
        behavior_confidence = float(analysis.get("behavior_confidence", 0.0))
    except (TypeError, ValueError):
        behavior_confidence = 0.0
    behavior_confidence = max(0.0, min(1.0, behavior_confidence))

    species_confidence = 0.0
    species_items = analysis.get("species")
    if isinstance(species_items, list):
        for item in species_items:
            if not isinstance(item, dict):
                continue
            if _normalize_species_name(item.get("name")).casefold() == primary_species.casefold():
                try:
                    species_confidence = float(item.get("confidence", 0.0))
                except (TypeError, ValueError):
                    species_confidence = 0.0
                break
    species_confidence = max(0.0, min(1.0, species_confidence))

    try:
        risk_score_10 = int(analysis.get("risk_score", 1))
    except (TypeError, ValueError):
        risk_score_10 = 1
    risk_score_10 = max(1, min(10, risk_score_10))
    risk_level = str(analysis.get("risk_level", "UNKNOWN")).strip().upper() or "UNKNOWN"
    if risk_level not in {"LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"}:
        risk_level = "UNKNOWN"

    human_present = bool(analysis.get("human_present", False))
    reasoning = str(analysis.get("risk_reasoning", "")).strip()
    recommendation = str(analysis.get("action_recommendation", "")).strip()
    uncertainty = str(analysis.get("uncertainty", "")).strip()

    risk = {
        "risk_score": round(risk_score_10 / 10.0, 4),
        "risk_level": risk_level,
        "gemini_risk_score": risk_score_10,
        "factors": [
            {"name": "human_presence", "value": human_present, "contribution": "Observed in video"},
            {"name": "risk_reasoning", "value": reasoning, "contribution": reasoning or "Not provided"},
            {"name": "action_recommendation", "value": recommendation, "contribution": recommendation or "Not provided"},
            {"name": "uncertainty", "value": uncertainty, "contribution": uncertainty or "Not provided"},
        ],
        "reasoning": reasoning,
        "action_recommendation": recommendation,
        "uncertainty": uncertainty,
    }

    successful = str(analysis.get("status", "success")).lower() == "success"
    has_subject = successful and primary_species != "UNKNOWN"
    model_version = analysis.get("model") or "unknown"

    behavior_entry = {
        "behaviour": primary_behavior,
        "behavior_class": primary_behavior.upper().replace(" ", "_"),
        "confidence": behavior_confidence,
        "frames": 0,
        "model_version": model_version,
        "behavior_timeline": analysis.get("behaviors", []),
        "source": "video_analysis",
    }
    species_entry = {
        "species": primary_species,
        "confidence": species_confidence,
        "sample_count": 0,
        "classified_count": 0,
        "source": "video_analysis",
    }
    risk_event = {
        "risk_event_id": f"video-{video.stem}",
        "track_id": "1",
        "species": primary_species,
        "behaviour": primary_behavior,
        "behaviour_confidence": behavior_confidence,
        "human_present": human_present,
        "risk": risk,
        "evidence_uri": None,
        "source": "video_analysis",
    }

    local_summary = local_result.get("summary", {}) if isinstance(local_result, dict) else {}
    local_models = local_result.get("models", {}) if isinstance(local_result, dict) else {}
    local_outputs = local_result.get("outputs", {}) if isinstance(local_result, dict) else {}

    local_pipeline_path = output_dir / "local_pipeline.json"
    if local_result is not None:
        local_pipeline_path.write_text(json.dumps(local_result, indent=2), encoding="utf-8")

    summary = dict(local_summary) if isinstance(local_summary, dict) else {}
    summary["unique_track_ids"] = ["1"] if has_subject else []
    summary["track_observation_count"] = 1 if has_subject else 0
    summary["detection_count"] = 1 if has_subject else 0

    result = {
        "input": str(video),
        "summary": summary,
        "species": {"1": species_entry} if has_subject else {},
        "behavior": {"1": behavior_entry} if has_subject else {},
        "risk_events": [risk_event] if has_subject else [],
        "models": {
            "detector": local_models.get("detector", "MegaDetectorV6 MDV6-yolov9-c"),
            "tracker": local_models.get("tracker", "ByteTrack"),
            "species": local_models.get("species", "SpeciesNet 5.x"),
            "behavior": model_version,
            "behavior_backend_error": None,
            "device": local_models.get("device", "cpu"),
            "authoritative_source": "video_analysis",
        },
        "outputs": {
            "annotated_video": local_outputs.get("annotated_video"),
            "tracks": local_outputs.get("tracks"),
            "species": local_outputs.get("species"),
            "crops": local_outputs.get("crops"),
            "behavior_clips": local_outputs.get("behavior_clips"),
            "evidence": local_outputs.get("evidence"),
            "gemini_analysis": str(output_dir / "gemini" / "analysis.json"),
            "local_pipeline": str(local_pipeline_path) if local_result is not None else None,
        },
        "gemini": analysis,
    }

    (output_dir / "pipeline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_authoritative(
    video: Path,
    output_dir: Path,
    sample_every: int = 3,
    species_samples: int = 16,
    behavior_checkpoint: str | Path = "models/behavior/videomae/videomae_combined_v1.pt",
) -> dict[str, Any]:
    """Run local diagnostics while making video-model output authoritative."""
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "gemini" / "analysis.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-diagnostics") as executor:
        local_future = executor.submit(
            run_integrated,
            video,
            output_dir,
            sample_every,
            species_samples,
            behavior_checkpoint,
        )

        try:
            analysis = analyze_video(video, output_path=analysis_path)
        except AnalysisUnavailableError:
            # Keep provider details out of the terminal. The API turns this into
            # a clean service-unavailable response rather than a fake UNKNOWN run.
            try:
                local_future.result()
            except Exception as exc:
                print(f"WARNING: local diagnostic pipeline failed: {type(exc).__name__}: {exc}")
            raise

        try:
            local_result = local_future.result()
        except Exception as exc:
            print(f"WARNING: local diagnostic pipeline failed: {type(exc).__name__}: {exc}")
            local_result = None

    return _dashboard_result(analysis, local_result, video, output_dir)
