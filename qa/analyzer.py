"""Visual QA via GPT-4o vision.

The analyzer focuses on what only vision + text can decide: object placement,
spatial coherence, irrelevant objects, on-screen text, brand mixing. It does
NOT decide the final status/score — the aggregator combines its output with
pre-computed audio-language and scene-similarity signals.

OpenAI payload is kept lean to control cost:
- Representative frames only (dense frames stay local for hashing).
- Capped at _MAX_FRAMES; long side downscaled to _MAX_IMAGE_LONG_SIDE.
- detail='low' so each image costs the flat ~85 tokens.
- Text fields are truncated and serialized without indent.
- Payload size is logged before the call; usage is logged after.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path

from openai import OpenAI
from PIL import Image


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


_LOG = logging.getLogger(__name__)

# Vision payload caps. detail='low' processes a 512px thumbnail at ~85 tokens
# regardless of source size, so going above ~768 buys nothing token-wise; the
# cap mainly cuts upload bytes and prompt-validation overhead.
_MAX_IMAGE_LONG_SIDE = 768
_IMAGE_JPEG_QUALITY = 75
_MAX_FRAMES = 6
_STT_MAX_CHARS = 3000
_OPTIONAL_TEXT_MAX_CHARS = 1500


def _encode_image_for_vision(path: Path, max_long_side: int = _MAX_IMAGE_LONG_SIDE) -> bytes:
    """Downscale a frame to max_long_side and return JPEG bytes."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        long_side = max(w, h)
        if long_side > max_long_side:
            scale = max_long_side / long_side
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=_IMAGE_JPEG_QUALITY, optimize=True)
        return buf.getvalue()


def _truncate(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + f"\n...[truncated, {len(t) - max_chars} chars omitted]"


def _compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _drop_empty(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if v not in (None, "", [], {})}


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

    frames = list(frame_paths)[:_MAX_FRAMES]

    user_blocks: list[dict] = [
        {"type": "text", "text": "## CLIENT INFO\n" + _compact_json(_drop_empty(client_info))},
        {"type": "text", "text": "## QA CONTEXT\n" + _compact_json(_drop_empty(qa_context))},
        {"type": "text", "text": f"## STT (lang={stt_language})\n{_truncate(stt_text, _STT_MAX_CHARS) or '(empty)'}"},
        {"type": "text", "text": "## PRE-COMPUTED SIGNALS (authoritative)\n" + _compact_json(pre_signals or {})},
    ]

    for label, text in (
        ("SCRIPT (intended)", script),
        ("GENERATION PROMPT", generation_prompt),
        ("SCENES (text)", scenes_text),
        ("REFERENCES", references),
    ):
        t = _truncate(text, _OPTIONAL_TEXT_MAX_CHARS)
        if t:
            user_blocks.append({"type": "text", "text": f"## {label}\n{t}"})

    user_blocks.append({"type": "text", "text": f"## FRAMES ({len(frames)} samples, in time order)"})

    image_b64_bytes_total = 0
    for path, offset in frames:
        img_bytes = _encode_image_for_vision(path)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        image_b64_bytes_total += len(b64)
        user_blocks.append({"type": "text", "text": f"frame @ {offset:.2f}s"})
        user_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low",
            },
        })

    user_blocks.append({"type": "text", "text": OUTPUT_SCHEMA})

    text_chars = sum(len(b["text"]) for b in user_blocks if b["type"] == "text")
    image_count = sum(1 for b in user_blocks if b["type"] == "image_url")
    pre_msg = (
        f"[QA openai] before-request model={model} detail=low "
        f"images={image_count} max_long_side={_MAX_IMAGE_LONG_SIDE} "
        f"text_chars={text_chars} image_b64_bytes={image_b64_bytes_total}"
    )
    _LOG.info(pre_msg)
    print(pre_msg)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_blocks},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    try:
        u = resp.usage
        post_msg = (
            f"[QA openai] after-response model={model} "
            f"prompt_tokens={getattr(u, 'prompt_tokens', '?')} "
            f"completion_tokens={getattr(u, 'completion_tokens', '?')} "
            f"total_tokens={getattr(u, 'total_tokens', '?')}"
        )
        _LOG.info(post_msg)
        print(post_msg)
    except Exception as e:
        print(f"[QA openai] usage log failed: {type(e).__name__}: {e}")

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
