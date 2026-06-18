"""Multimodal QA judgment via GPT-4o (vision + text). Returns dict matching qa_result schema."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = """You are a strict QA reviewer for short marketing videos produced for Korean small-business clients.
You receive:
- CLIENT INFO (the only source of truth about the brand, services, claims allowed)
- STT TEXT (what the video's voice actually says) + detected language code
- SCRIPT (optional, the intended narration)
- SCENES / GENERATION CONTEXT (optional)
- 6 sampled FRAMES from the video, in time order

Your job is to detect mismatches against CLIENT INFO and output a single JSON verdict.
Be skeptical: if a number, certification, price, location, or success rate is not stated
in client_info, treat it as an unsupported claim. Do not invent positives."""

RULES = """DECISION RULES (apply in order):
1) Other company / brand name detected anywhere (STT or on-screen) that is not the client
   -> status=retry, detectedCompanyMixing=true, criticalIssue describing which name.
2) Claim with specifics (results, %, numbers, awards, certifications, prices, exact location)
   not present in client_info
   -> status=retry, detectedUnsupportedClaim=true.
3) STT language is not 'ko' (Korean) OR voice is clearly non-Korean
   -> status=retry, detectedForeignLanguage=true.
4) Industry mismatch — scenes show a totally different business type vs client_info.industry
   -> status=retry, detectedWrongIndustry=true.
5) Medical / legal / financial / income-related exaggeration, OR client_info.forbiddenClaims
   present in STT or frames
   -> status=human_review, humanReviewReason describing the risk.
6) Broken / corrupted subtitles or on-screen text severely cropped, unreadable, or overlapping
   -> status=human_review, detectedVisualTextIssue=true.
7) STT vs SCRIPT large divergence (skipped sentences, swapped meaning)
   -> detectedAudioScriptMismatch=true. If meaning changed in a risky way -> human_review,
   otherwise warning.
8) None of the above, only minor wording issues
   -> status=pass, list nits in warnings.

SCORING: 100=clean, 80-99=pass with warnings, 50-79=retry, <50=fail.
status=fail only when video is unusable (broken file, totally wrong client, no audio at all).

RETRY PROMPT: when status=retry you MUST write a concrete Korean retryPrompt that the user
can paste into the external video-generation platform to regenerate. Include:
- the correct client name + industry
- which services/promotionPoints to feature
- which forbidden claims to avoid
- specific fixes for what went wrong this time (no other brands, no unsupported specifics, etc.)
Keep retryPrompt under 600 chars."""

OUTPUT_SCHEMA = """OUTPUT a single JSON object, no markdown, no prose. Schema:
{
  "status": "pass" | "retry" | "human_review" | "fail",
  "score": 0..100,
  "criticalIssues": [string],
  "warnings": [string],
  "detectedForeignLanguage": bool,
  "detectedCompanyMixing": bool,
  "detectedUnsupportedClaim": bool,
  "detectedWrongIndustry": bool,
  "detectedVisualTextIssue": bool,
  "detectedAudioScriptMismatch": bool,
  "retryPrompt": string,
  "humanReviewReason": string
}"""


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def analyze(
    *,
    client_info: dict,
    stt_text: str,
    stt_language: str,
    frame_paths: list[tuple[Path, float]],
    script: str | None,
    scenes: list | dict | None,
    generation_context: dict | None,
    model: str = "gpt-4o",
) -> dict:
    client = _get_client()

    user_blocks: list[dict] = [
        {"type": "text", "text": "## CLIENT INFO\n" + json.dumps(client_info, ensure_ascii=False, indent=2)},
        {"type": "text", "text": f"## STT (whisper detected language='{stt_language}')\n{stt_text or '(empty)'}"},
    ]
    if script:
        user_blocks.append({"type": "text", "text": "## SCRIPT (intended)\n" + script})
    if scenes is not None:
        user_blocks.append({"type": "text", "text": "## SCENES\n" + json.dumps(scenes, ensure_ascii=False)})
    if generation_context is not None:
        user_blocks.append({"type": "text", "text": "## GENERATION CONTEXT\n" + json.dumps(generation_context, ensure_ascii=False)})

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

    user_blocks.append({"type": "text", "text": RULES})
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
