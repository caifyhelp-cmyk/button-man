"""Visual QA via GPT-4o vision.

The analyzer focuses on what only vision + text can decide: object placement,
spatial coherence, irrelevant objects, on-screen text, brand mixing. It does
NOT decide the final status/score — the aggregator combines its output with
pre-computed audio-language and scene-similarity signals.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """You are a multimodal QA reviewer for short marketing videos
produced by an external AI video platform. Your job is to LOOK at the supplied
frames and report visual/scene anomalies in JSON. You do NOT decide final
status/score — produce observations and visual flags only.

PRIMARY FOCUS — does the video look obviously wrong as a finished product?
1) Floating furniture/objects not resting on the floor; objects hovering
   without support.
2) Spatial distortion: broken wall/ceiling/floor/door/furniture proportions,
   collapsing perspective, melting or smearing object shapes, composition
   artifacts on object boundaries.
3) Objects unrelated to qa_context.sceneIntent / client_info.industry /
   qa_context.expectedSubjects, or anything in qa_context.forbiddenObjects.
4) On-screen text broken, cropped, unreadable, or overlapping.
5) Other-company brand names visible on screen.

You also receive PRE-COMPUTED SIGNALS (sceneDiversityScore, duplicateSceneRanges,
audioLanguageSummary). Treat those as authoritative — do NOT recompute them.
Use them only as context to corroborate other observations.

Be biased toward catching obviously-broken AI output rather than chasing
subtle nits. When a frame would clearly make a viewer say "this is AI-generated
and broken", flag it with severity 'high'."""


OUTPUT_SCHEMA = """OUTPUT a single JSON object, no markdown, no prose. Schema:
{
  "detectedFloatingObjects": bool,
  "detectedSpatialDistortion": bool,
  "detectedIrrelevantObjects": bool,
  "detectedVisualTextIssue": bool,
  "detectedCompanyMixing": bool,
  "detectedUnsupportedClaim": bool,
  "detectedWrongIndustry": bool,
  "detectedAudioScriptMismatch": bool,

  "visualAnomalyFrames": [
    {"offsetSec": number,
     "category": "floating_furniture"|"spatial_distortion"|"melting_shape"|"composition_artifact"|"broken_text"|"other",
     "severity": "low"|"medium"|"high",
     "description": string}
  ],
  "irrelevantObjectFindings": [
    {"offsetSec": number|null, "object": string, "expectedContext": string,
     "severity": "low"|"medium"|"high"}
  ],
  "visualQaSummary": [
    {"offsetSec": number, "observations": [string],
     "spatialOk": bool, "floatingObjects": [string], "score": number|null}
  ],
  "sceneQaSummary": [
    {"sceneIdx": number, "description": string, "expectedThemes": [string],
     "observedThemes": [string], "matchesIntent": bool|null, "note": string}
  ],
  "criticalIssues": [string],
  "warnings": [string]
}

Notes:
- Include one visualQaSummary entry per supplied frame.
- If you cannot decide spatialOk for a frame, set spatialOk=null.
- criticalIssues/warnings should be in Korean, since the operator reads them.
- score per frame is optional (0..100); set null if uncertain."""


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def run_visual_qa(
    *,
    client_info: dict,
    qa_context: dict,
    stt_text: str,
    stt_language: str,
    script: str | None,
    scenes_text: str | None,
    generation_prompt: str | None,
    references: str | None,
    pre_signals: dict,
    frame_paths: list[tuple[Path, float]],
    model: str = "gpt-4o",
) -> dict:
    client = _get_client()

    user_blocks: list[dict] = [
        {"type": "text", "text": "## CLIENT INFO\n" + json.dumps(client_info, ensure_ascii=False, indent=2)},
        {"type": "text", "text": "## QA CONTEXT\n" + json.dumps(qa_context, ensure_ascii=False, indent=2)},
        {"type": "text", "text": f"## STT (whisper detected language='{stt_language}')\n{stt_text or '(empty)'}"},
        {"type": "text", "text": "## PRE-COMPUTED SIGNALS (authoritative)\n" + json.dumps(pre_signals, ensure_ascii=False, indent=2)},
    ]
    if script:
        user_blocks.append({"type": "text", "text": "## SCRIPT (intended)\n" + script})
    if generation_prompt:
        user_blocks.append({"type": "text", "text": "## GENERATION PROMPT\n" + generation_prompt})
    if scenes_text:
        user_blocks.append({"type": "text", "text": "## SCENES (text)\n" + scenes_text})
    if references:
        user_blocks.append({"type": "text", "text": "## REFERENCES\n" + references})

    user_blocks.append({"type": "text", "text": f"## FRAMES ({len(frame_paths)} samples, in time order)"})
    for path, offset in frame_paths:
        user_blocks.append({"type": "text", "text": f"frame @ {offset:.2f}s"})
        user_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{_encode_image(path)}",
                "detail": "low",
            },
        })

    user_blocks.append({"type": "text", "text": OUTPUT_SCHEMA})

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_blocks},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or "{}"
    return json.loads(raw)


def analyze(
    *,
    client_info: dict,
    stt_text: str,
    stt_language: str,
    frame_paths: list[tuple[Path, float]],
    script: str | None,
    scenes,
    generation_context,
    model: str = "gpt-4o",
) -> dict:
    """Backward-compat wrapper for callers that haven't switched to run_visual_qa."""
    return run_visual_qa(
        client_info=client_info,
        qa_context={},
        stt_text=stt_text,
        stt_language=stt_language,
        script=script,
        scenes_text=json.dumps(scenes, ensure_ascii=False) if scenes else None,
        generation_prompt=json.dumps(generation_context, ensure_ascii=False) if generation_context else None,
        references=None,
        pre_signals={},
        frame_paths=frame_paths,
        model=model,
    )
