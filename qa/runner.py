"""Orchestrator: extract audio + frames, run STT, call analyzer, write qa_result.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .analyzer import analyze
from .ffmpeg_utils import capture_frames, extract_audio
from .stt import transcribe

load_dotenv()

SAMPLE_OFFSETS_SEC = [0, 2, 5, 10, 15]


def _load_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def run_qa(job_id: str, inputs_dir: str = "qa-inputs") -> dict:
    job_dir = Path(inputs_dir) / job_id
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job folder not found: {job_dir}")

    video = job_dir / "video.mp4"
    client_info_path = job_dir / "client_info.json"
    if not video.exists():
        raise FileNotFoundError(f"Missing required file: {video}")
    if not client_info_path.exists():
        raise FileNotFoundError(f"Missing required file: {client_info_path}")

    client_info = json.loads(client_info_path.read_text(encoding="utf-8"))
    script = _load_optional_text(job_dir / "script.txt")
    scenes = _load_optional_json(job_dir / "scenes.json")
    gen_ctx = _load_optional_json(job_dir / "generation_context.json")

    artifacts = job_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)

    audio_path = extract_audio(video, artifacts / "audio.mp3")
    frames = capture_frames(video, artifacts, SAMPLE_OFFSETS_SEC)
    stt = transcribe(audio_path)

    analysis = analyze(
        client_info=client_info,
        stt_text=stt["text"],
        stt_language=stt["language"],
        frame_paths=frames,
        script=script,
        scenes=scenes,
        generation_context=gen_ctx,
    )

    result = {
        "jobId": job_id,
        "status": analysis.get("status", "human_review"),
        "score": int(analysis.get("score", 0)),
        "criticalIssues": analysis.get("criticalIssues", []),
        "warnings": analysis.get("warnings", []),
        "detectedForeignLanguage": bool(analysis.get("detectedForeignLanguage", False)),
        "detectedCompanyMixing": bool(analysis.get("detectedCompanyMixing", False)),
        "detectedUnsupportedClaim": bool(analysis.get("detectedUnsupportedClaim", False)),
        "detectedWrongIndustry": bool(analysis.get("detectedWrongIndustry", False)),
        "detectedVisualTextIssue": bool(analysis.get("detectedVisualTextIssue", False)),
        "detectedAudioScriptMismatch": bool(analysis.get("detectedAudioScriptMismatch", False)),
        "detectedFloatingObjects": bool(analysis.get("detectedFloatingObjects", False)),
        "detectedSpatialDistortion": bool(analysis.get("detectedSpatialDistortion", False)),
        "detectedIrrelevantObjects": bool(analysis.get("detectedIrrelevantObjects", False)),
        "detectedDuplicateScenes": bool(analysis.get("detectedDuplicateScenes", False)),
        "detectedForeignLanguageTTS": bool(
            analysis.get("detectedForeignLanguageTTS",
                         analysis.get("detectedForeignLanguage", False))
        ),
        "sttText": stt["text"],
        "sttLanguage": stt["language"],
        "frameSummary": [
            {"path": str(p.relative_to(job_dir)).replace("\\", "/"),
             "offsetSec": round(t, 2)}
            for p, t in frames
        ],
        "sceneDiversityScore": analysis.get("sceneDiversityScore"),
        "duplicateSceneRanges": analysis.get("duplicateSceneRanges", []),
        "visualAnomalyFrames": analysis.get("visualAnomalyFrames", []),
        "irrelevantObjectFindings": analysis.get("irrelevantObjectFindings", []),
        "audioLanguageSummary": analysis.get("audioLanguageSummary", {
            "primary": stt["language"], "confidence": None,
            "detectedSecondary": [], "foreignSegments": [],
        }),
        "visualQaSummary": analysis.get("visualQaSummary", []),
        "sceneQaSummary": analysis.get("sceneQaSummary", []),
        "retryPrompt": analysis.get("retryPrompt", ""),
        "humanReviewReason": analysis.get("humanReviewReason", ""),
        "checkedAt": datetime.now(timezone.utc).isoformat(),
    }

    out = job_dir / "qa_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
