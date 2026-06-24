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

SECONDARY FOCUS — AI-generation-specific defects and marketing-claim risks.
Report each item below as a structured entry inside `detailedFindings`. Use
the schema in OUTPUT_SCHEMA. Only emit a finding when you can point to a real
observation; do NOT fabricate evidence.

A) subtitle_narration_mismatch — on-screen subtitle text differs from the
   spoken narration in the STT. Minor wording variance = low. A meaning
   change (different numbers, different product names, opposite claims) = high.
B) brand_name_misuse — the client's brand / company / service name appears
   misspelled, abbreviated wrong, or rendered as a similar-but-different
   name. This is distinct from "other-company mixing" — here the name is
   meant to be the client's but is rendered wrong.
C) logo_text_corruption — logos look melted/smeared/duplicated, or on-screen
   text reads as gibberish / glyph soup / unreadable foreign letters. This
   is a stricter sub-case of on-screen text issue; flag here when it is
   clearly an AI-generation artifact, not just a font/crop issue.
D) human_body_distortion — wrong finger count, extra limbs, melted hands,
   warped faces, distorted eyes/mouth — typical generative-AI body errors.
E) scene_industry_mismatch — the WHOLE scene's setting/activity does not
   match the client's industry/product/service (not a single stray object,
   but the overall scene). e.g., a hospital interior in a coffee-shop ad.
F) exaggerated_claim — narration or subtitles use unprovable absolutes:
   "무조건", "100%", "반드시", "최고", "완벽", "억대 수익", "확실한 효과".
   Cross-check client_info.forbiddenClaims; treat overlap as high severity.
G) authority_claim_risk — claims of government / agency / certification
   backing ("정부 인증", "공식 지정", "고용노동부 인정", "공단 공식",
   "법정 필수", "국가 지원"). Mark as high unless the client info clearly
   supports the claim (industry/services/promotionPoints).
H) aggressive_cta — high-pressure CTA: "지금 안 하면 손해", "오늘만",
   "무조건 신청", "당장 구매". Pure urgency words alone = medium; combined
   with loss-framing / scarcity = high.
I) unclear_message — the video has no voiceover and no subtitles, OR the
   core message is so unclear that a viewer cannot tell what is being
   promoted. low if just hard to follow; high if no message at all.
J) pacing_issue — quality category, NOT legal risk: excessive repetition
   of one scene, one scene held too long, OR a shorts-format video that
   does not surface its hook in the early seconds. Severity always <= medium.

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

  "detectedSubtitleNarrationMismatch": bool,
  "detectedBrandNameMisuse": bool,
  "detectedLogoTextCorruption": bool,
  "detectedHumanBodyDistortion": bool,
  "detectedSceneIndustryMismatch": bool,
  "detectedExaggeratedClaim": bool,
  "detectedAuthorityClaimRisk": bool,
  "detectedAggressiveCta": bool,
  "detectedUnclearMessage": bool,
  "detectedPacingIssue": bool,

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

  "detailedFindings": [
    {"type": "subtitle_narration_mismatch"|"brand_name_misuse"|"logo_text_corruption"|
             "human_body_distortion"|"scene_industry_mismatch"|"exaggerated_claim"|
             "authority_claim_risk"|"aggressive_cta"|"unclear_message"|"pacing_issue",
     "detected": bool,
     "severity": "low"|"medium"|"high",
     "reason": string,
     "timeRange": string|null,
     "evidence": string|null,
     "suggestion": string|null}
  ],

  "criticalIssues": [string],
  "warnings": [string]
}

Notes:
- Include one visualQaSummary entry per supplied frame.
- If you cannot decide spatialOk for a frame, set spatialOk=null.
- criticalIssues/warnings should be in Korean, since the operator reads them.
- score per frame is optional (0..100); set null if uncertain.
- detailedFindings: include an entry ONLY when detected=true. Each `reason`,
  `evidence`, `suggestion` should be in Korean (operator reads them). `timeRange`
  uses formats like "0:00-0:03" or "@1:24" or null if not localizable.
- severity rules:
    high   = legal/brand risk OR video should not be used as-is
    medium = needs reviewer fix before publishing
    low    = informational / quality nit (pacing_issue caps at medium)
- The boolean `detected*` flags MUST agree with the corresponding entries in
  detailedFindings: if a finding is present with detected=true, set the flag
  true; otherwise false."""


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
            },
        })

    user_blocks.append({"type": "text", "text": OUTPUT_SCHEMA})

    # Pre-request payload size log (sizes/counts only; no raw base64, no API key).
    text_chars = sum(len(b["text"]) for b in user_blocks if b["type"] == "text")
    image_count = sum(1 for b in user_blocks if b["type"] == "image_url")
    image_url_chars = sum(len(b["image_url"]["url"]) for b in user_blocks if b["type"] == "image_url")
    print(
        f"[QA openai] before-request model={model} images={image_count} "
        f"text_chars={text_chars} image_url_chars={image_url_chars}"
    )

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
        print(
            f"[QA openai] after-response model={model} "
            f"prompt_tokens={getattr(u, 'prompt_tokens', '?')} "
            f"completion_tokens={getattr(u, 'completion_tokens', '?')} "
            f"total_tokens={getattr(u, 'total_tokens', '?')}"
        )
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
