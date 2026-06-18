"""End-to-end QA pipeline for the /api/qa/run multipart endpoint.

Service-layer composition:
  ffmpeg_utils.extract_audio
  stt.transcribe_audio              (OpenAI Whisper)
  language.detect_audio_language
  ffmpeg_utils.capture_frames       (representative, for vision)
  ffmpeg_utils.capture_dense_frames (small, for hashing only)
  similarity.calculate_frame_similarity
  similarity.detect_duplicate_scenes
  analyzer.run_visual_qa            (GPT-4o vision)
  aggregator.aggregate_qa_results

Each step can be swapped independently — change the import and the function
contract stays the same.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .aggregator import aggregate_qa_results
from .analyzer import run_visual_qa
from .ffmpeg_utils import (
    capture_dense_frames,
    capture_frames,
    extract_audio,
    frame_to_data_url,
)
from .language import detect_audio_language
from .similarity import calculate_frame_similarity, detect_duplicate_scenes
from .stt import transcribe_audio


def run_qa_on_upload(
    *,
    video_bytes: bytes,
    video_filename: str,
    client_info: dict,
    qa_context: dict,
    script: str | None,
    scenes_text: str | None,
    generation_prompt: str | None,
    references: str | None,
) -> dict[str, Any]:
    """Full real-mode pipeline. Returns the qa_result dict including data-URL frames."""
    with tempfile.TemporaryDirectory(prefix="qa_") as td:
        td_path = Path(td)
        video = td_path / "video.mp4"
        video.write_bytes(video_bytes)

        artifacts = td_path / "artifacts"

        # 1) audio extract → STT → language detect
        audio_path = extract_audio(video, artifacts / "audio.mp3")
        stt_result = transcribe_audio(audio_path)
        audio_language = detect_audio_language(stt_result)

        # 2) representative frames (sent to vision AND shown in UI as data URLs)
        rep_frames = capture_frames(video, artifacts)
        frame_summary = [
            {
                "offsetSec": round(t, 2),
                "imageDataUrl": frame_to_data_url(p),
            }
            for p, t in rep_frames
        ]

        # 3) dense frames → dHash → cluster → duplicate ranges + diversity
        dense_frames = capture_dense_frames(video, artifacts, target_count=30)
        hashes = calculate_frame_similarity(dense_frames)
        dup_ranges, diversity_score = detect_duplicate_scenes(hashes)
        similarity_result = {
            "duplicateSceneRanges": dup_ranges,
            "sceneDiversityScore": diversity_score,
        }

        # 4) visual QA via vision LLM (representative frames only)
        visual_analysis = run_visual_qa(
            client_info=client_info,
            qa_context=qa_context,
            stt_text=stt_result.get("text") or "",
            stt_language=audio_language["primary"],
            script=script,
            scenes_text=scenes_text,
            generation_prompt=generation_prompt,
            references=references,
            pre_signals={
                "sceneDiversityScore": diversity_score,
                "duplicateSceneRanges": dup_ranges,
                "audioLanguageSummary": audio_language,
            },
            frame_paths=rep_frames,
        )

        # 5) aggregate all signals → final status/score/retryPrompt
        result = aggregate_qa_results(
            client_info=client_info,
            qa_context=qa_context,
            video_meta={
                "filename": video_filename,
                "size": len(video_bytes),
            },
            stt_result=stt_result,
            audio_language=audio_language,
            similarity_result=similarity_result,
            visual_analysis=visual_analysis,
        )

        result["frameSummary"] = frame_summary
        result["_meta"] = {
            "mode": "real",
            "video": {"filename": video_filename, "size": len(video_bytes)},
            "denseFramesAnalyzed": len(dense_frames),
            "representativeFramesAnalyzed": len(rep_frames),
            "sttSegments": len(stt_result.get("segments") or []),
            "sttModel": stt_result.get("_model"),
        }
        return result
