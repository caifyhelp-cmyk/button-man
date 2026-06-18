"""CLI orchestrator — same pipeline as web_runner but reads from qa-inputs/{jobId}/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .aggregator import aggregate_qa_results
from .analyzer import run_visual_qa
from .ffmpeg_utils import capture_dense_frames, capture_frames, extract_audio
from .language import detect_audio_language
from .similarity import calculate_frame_similarity, detect_duplicate_scenes
from .stt import transcribe_audio

load_dotenv()


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
    qa_context = _load_optional_json(job_dir / "qa_context.json") or {}
    script = _load_optional_text(job_dir / "script.txt")
    scenes_text = _load_optional_text(job_dir / "scenes.json")
    generation_prompt = _load_optional_text(job_dir / "generation_prompt.txt")

    artifacts = job_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)

    audio_path = extract_audio(video, artifacts / "audio.mp3")
    stt_result = transcribe_audio(audio_path)
    audio_language = detect_audio_language(stt_result)

    rep_frames = capture_frames(video, artifacts)
    dense_frames = capture_dense_frames(video, artifacts, target_count=30)
    hashes = calculate_frame_similarity(dense_frames)
    dup_ranges, diversity_score = detect_duplicate_scenes(hashes)

    visual = run_visual_qa(
        client_info=client_info,
        qa_context=qa_context,
        stt_text=stt_result.get("text") or "",
        stt_language=audio_language["primary"],
        script=script,
        scenes_text=scenes_text,
        generation_prompt=generation_prompt,
        references=None,
        pre_signals={
            "sceneDiversityScore": diversity_score,
            "duplicateSceneRanges": dup_ranges,
            "audioLanguageSummary": audio_language,
        },
        frame_paths=rep_frames,
    )

    result = aggregate_qa_results(
        client_info=client_info,
        qa_context=qa_context,
        video_meta={"filename": "video.mp4", "size": video.stat().st_size},
        stt_result=stt_result,
        audio_language=audio_language,
        similarity_result={
            "duplicateSceneRanges": dup_ranges,
            "sceneDiversityScore": diversity_score,
        },
        visual_analysis=visual,
    )
    result["jobId"] = job_id
    result["frameSummary"] = [
        {"path": str(p.relative_to(job_dir)).replace("\\", "/"),
         "offsetSec": round(t, 2)}
        for p, t in rep_frames
    ]
    result["_meta"] = {
        "mode": "real",
        "denseFramesAnalyzed": len(dense_frames),
        "representativeFramesAnalyzed": len(rep_frames),
        "sttSegments": len(stt_result.get("segments") or []),
    }

    out = job_dir / "qa_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
