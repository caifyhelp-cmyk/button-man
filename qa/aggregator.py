"""Combine visual + audio + similarity signals into final QA verdict.

The visual analyzer (LLM) outputs visual-only flags and observations. This
aggregator merges those with pre-computed scene-similarity and audio-language
signals and decides the final status, score, retryPrompt, humanReviewReason.

Final result dict matches the qa_result schema published in qa/README.md.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# Display labels for detailedFindings entry types (Korean — operator-facing).
FINDING_LABELS: dict[str, str] = {
    "subtitle_narration_mismatch": "자막-나레이션 불일치",
    "brand_name_misuse": "브랜드명/상호명 오기재",
    "logo_text_corruption": "로고/화면 텍스트 깨짐",
    "human_body_distortion": "인물/손/얼굴 왜곡",
    "scene_industry_mismatch": "제품/서비스와 무관한 장면",
    "exaggerated_claim": "과장·보장 표현",
    "authority_claim_risk": "공식기관/법령/인증 표현 리스크",
    "aggressive_cta": "CTA 과도함",
    "unclear_message": "정보 전달 불가",
    "pacing_issue": "영상 길이/씬 구성 이상",
}

# pacing_issue is quality-only; do not let it push the overall verdict to "hold".
_QUALITY_ONLY_TYPES: set[str] = {"pacing_issue"}


def _normalize_detailed_findings(raw: Any) -> list[dict[str, Any]]:
    """Keep only entries with detected=true and valid type; coerce missing fields."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if not item.get("detected"):
            continue
        ftype = item.get("type")
        if ftype not in FINDING_LABELS:
            continue
        sev = item.get("severity")
        if sev not in ("low", "medium", "high"):
            sev = "medium"
        out.append({
            "type": ftype,
            "label": FINDING_LABELS[ftype],
            "detected": True,
            "severity": sev,
            "reason": (item.get("reason") or "").strip(),
            "timeRange": item.get("timeRange") or None,
            "evidence": item.get("evidence") or None,
            "suggestion": item.get("suggestion") or None,
        })
    return out


_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


def _build_report(*, legacy_status: str, detailed_findings: list[dict]) -> dict[str, Any]:
    """Derive operator-facing summary from legacy status + new findings.

    Does NOT call an LLM and does NOT introduce a new judging stage — it is a
    deterministic relabel of signals already produced upstream.
    """
    risk_findings = [f for f in detailed_findings if f["type"] not in _QUALITY_ONLY_TYPES]
    high_findings = [f for f in risk_findings if f["severity"] == "high"]
    medium_findings = [f for f in risk_findings if f["severity"] == "medium"]
    major_count = len(high_findings) + len(medium_findings)

    # Verdict mapping: never softens legacy status, only escalates when new
    # high-severity findings appear that legacy logic would not have caught.
    if legacy_status in ("retry", "fail"):
        verdict = "hold"
    elif legacy_status == "human_review":
        verdict = "review"
    else:  # legacy_status == 'pass' (or unknown — be conservative)
        if len(high_findings) >= 2:
            verdict = "hold"
        elif high_findings:
            verdict = "review"
        elif medium_findings:
            verdict = "review"
        else:
            verdict = "pass"

    sorted_findings = sorted(
        detailed_findings,
        key=lambda f: (
            -_SEVERITY_RANK.get(f["severity"], 0),
            0 if f["type"] not in _QUALITY_ONLY_TYPES else 1,
        ),
    )
    top_priority = sorted_findings[:3]

    suggestions: list[str] = []
    seen: set[str] = set()
    for f in sorted_findings:
        s = (f.get("suggestion") or "").strip()
        if not s or s in seen:
            continue
        suggestions.append(s)
        seen.add(s)
        if len(suggestions) >= 5:
            break

    return {
        "overallVerdict": verdict,
        "verdictLabel": {"pass": "통과", "review": "검토 필요", "hold": "사용 보류"}[verdict],
        "majorIssueCount": major_count,
        "highIssueCount": len(high_findings),
        "topPriority": top_priority,
        "suggestions": suggestions,
    }


def _build_retry_prompt(
    client_info: dict,
    qa_context: dict,
    critical: list[str],
) -> str:
    name = client_info.get("clientName") or "(고객사명 미입력)"
    industry = client_info.get("industry") or "(업종 미입력)"
    services = client_info.get("services") or []
    promo = client_info.get("promotionPoints") or []
    forbidden = client_info.get("forbiddenClaims") or []
    tone = client_info.get("brandTone") or ""

    intent = qa_context.get("sceneIntent") or ""
    expected = qa_context.get("expectedSubjects") or []
    fobj = qa_context.get("forbiddenObjects") or []
    vtype = qa_context.get("videoType") or ""

    lines = [f"[재생성 요청] {name} ({industry})", ""]
    if vtype:
        lines.append(f"영상 유형: {vtype}")
    if intent:
        lines.append(f"씬 의도: {intent}")
    if vtype or intent:
        lines.append("")
    lines.append("## 이번 영상에서 수정할 점")
    if critical:
        lines.extend(f"- {c}" for c in critical)
    else:
        lines.append("- (자동 감지된 치명 문제 없음 — 사람 검수 사유 별도 확인)")

    lines.extend(["", "## 반드시 지킬 것", f"- 고객사명: {name}", f"- 업종: {industry}"])
    if services:
        lines.append(f"- 다룰 서비스: {', '.join(services[:5])}")
    if promo:
        lines.append(f"- 홍보 포인트: {', '.join(promo[:3])}")
    if expected:
        lines.append(f"- 나와야 할 주요 대상: {', '.join(expected[:6])}")
    if fobj:
        lines.append(f"- 나오면 안 되는 물체: {', '.join(fobj)}")
    if forbidden:
        lines.append(f"- 금지 표현 (절대 사용 금지): {', '.join(forbidden)}")
    if tone:
        lines.append(f"- 브랜드 톤: {tone}")
    lines.extend([
        "",
        "## 시각 / 장면 원칙",
        "- 공중에 떠 있거나 바닥에 자연스럽게 놓이지 않은 가구·물체가 없어야 합니다.",
        "- 벽·천장·바닥·문틀·가구 비율의 비정상적 왜곡이 없어야 합니다.",
        "- 형태가 녹아내리거나 뭉개진 물체가 없어야 합니다.",
        f"- {industry or '해당 업종'}/씬 의도와 무관한 객체가 등장하지 않아야 합니다.",
        "- 같은 장면이 반복되지 않고 씬 다양성을 확보해야 합니다.",
        "",
        "## 오디오 원칙",
        "- 한국어 TTS만 사용합니다. 외국어 음성이 섞이지 않아야 합니다.",
        "- client_info에 없는 수치·인증·가격·위치는 단정하지 않습니다.",
    ])
    return "\n".join(lines)


def aggregate_qa_results(
    *,
    client_info: dict,
    qa_context: dict,
    video_meta: dict,
    stt_result: dict,
    audio_language: dict,
    similarity_result: dict,
    visual_analysis: dict,
) -> dict[str, Any]:
    critical: list[str] = list(visual_analysis.get("criticalIssues") or [])
    warnings: list[str] = list(visual_analysis.get("warnings") or [])

    sim_ranges = similarity_result.get("duplicateSceneRanges") or []
    sim_score = similarity_result.get("sceneDiversityScore")
    foreign_segments = audio_language.get("foreignSegments") or []
    primary_lang = (audio_language.get("primary") or "unknown").lower()
    # Foreign-language risk requires confirmed human speech AND that we
    # actually used the language-detection output. For BGM/silence/SFX we
    # never penalize foreign labels — Whisper mis-tags noise as Khmer/Thai/etc.
    speech_present = bool(audio_language.get("speechPresent"))
    language_detection_used = bool(audio_language.get("languageDetectionUsed"))
    has_transcribed_text = bool((stt_result.get("text") or "").strip())
    lang_confidence = audio_language.get("confidence") or 0.0
    foreign_lang_eligible = (
        speech_present
        and language_detection_used
        and has_transcribed_text
        and lang_confidence >= 0.5
    )
    foreign_lang_tts = foreign_lang_eligible and (
        (primary_lang not in ("ko", "unknown")) or bool(foreign_segments)
    )

    visual_anomaly_frames = visual_analysis.get("visualAnomalyFrames") or []
    irrelevant_findings = visual_analysis.get("irrelevantObjectFindings") or []

    high_floating = [
        f for f in visual_anomaly_frames
        if f.get("category") == "floating_furniture" and f.get("severity") in ("high", "medium")
    ]
    high_distortion = [
        f for f in visual_anomaly_frames
        if f.get("category") in ("spatial_distortion", "melting_shape")
           and f.get("severity") == "high"
    ]

    detailed_findings = _normalize_detailed_findings(visual_analysis.get("detailedFindings"))
    findings_by_type: dict[str, dict] = {f["type"]: f for f in detailed_findings}

    flags = {
        "detectedFloatingObjects": bool(visual_analysis.get("detectedFloatingObjects")) or bool(high_floating),
        "detectedSpatialDistortion": bool(visual_analysis.get("detectedSpatialDistortion")) or bool(high_distortion),
        "detectedIrrelevantObjects": bool(visual_analysis.get("detectedIrrelevantObjects")) or bool(irrelevant_findings),
        "detectedDuplicateScenes": bool(
            (sim_score is not None and sim_score < 70) and len(sim_ranges) > 0
        ),
        "detectedForeignLanguageTTS": foreign_lang_tts,
        "detectedForeignLanguage": foreign_lang_tts,
        "detectedCompanyMixing": bool(visual_analysis.get("detectedCompanyMixing")),
        "detectedUnsupportedClaim": bool(visual_analysis.get("detectedUnsupportedClaim")),
        "detectedWrongIndustry": bool(visual_analysis.get("detectedWrongIndustry")),
        "detectedVisualTextIssue": bool(visual_analysis.get("detectedVisualTextIssue")),
        "detectedAudioScriptMismatch": bool(visual_analysis.get("detectedAudioScriptMismatch")),
    }

    # New-category flags. Booleans from the model are accepted, and we also
    # raise the flag whenever the structured detailedFindings array carries
    # the matching type — so the UI stays consistent even if the LLM only
    # populated one of the two surfaces.
    enhanced_flags = {
        "detectedSubtitleNarrationMismatch": bool(visual_analysis.get("detectedSubtitleNarrationMismatch"))
            or ("subtitle_narration_mismatch" in findings_by_type),
        "detectedBrandNameMisuse": bool(visual_analysis.get("detectedBrandNameMisuse"))
            or ("brand_name_misuse" in findings_by_type),
        "detectedLogoTextCorruption": bool(visual_analysis.get("detectedLogoTextCorruption"))
            or ("logo_text_corruption" in findings_by_type),
        "detectedHumanBodyDistortion": bool(visual_analysis.get("detectedHumanBodyDistortion"))
            or ("human_body_distortion" in findings_by_type),
        "detectedSceneIndustryMismatch": bool(visual_analysis.get("detectedSceneIndustryMismatch"))
            or ("scene_industry_mismatch" in findings_by_type),
        "detectedExaggeratedClaim": bool(visual_analysis.get("detectedExaggeratedClaim"))
            or ("exaggerated_claim" in findings_by_type),
        "detectedAuthorityClaimRisk": bool(visual_analysis.get("detectedAuthorityClaimRisk"))
            or ("authority_claim_risk" in findings_by_type),
        "detectedAggressiveCta": bool(visual_analysis.get("detectedAggressiveCta"))
            or ("aggressive_cta" in findings_by_type),
        "detectedUnclearMessage": bool(visual_analysis.get("detectedUnclearMessage"))
            or ("unclear_message" in findings_by_type),
        "detectedPacingIssue": bool(visual_analysis.get("detectedPacingIssue"))
            or ("pacing_issue" in findings_by_type),
    }

    # Inject signal-derived findings
    if sim_score is not None:
        if sim_score < 40 and sim_ranges:
            critical.append(
                f"씬 다양성이 매우 낮습니다 (점수 {sim_score}/100, 중복 그룹 {len(sim_ranges)}개)."
            )
        elif sim_score < 70 and sim_ranges:
            warnings.append(
                f"중복 씬 구간 발견 (다양성 {sim_score}/100, 그룹 {len(sim_ranges)}개)."
            )
    if foreign_lang_eligible:
        if primary_lang and primary_lang not in ("ko", "unknown"):
            critical.append(f"오디오 주 언어가 한국어가 아닙니다 (감지: {primary_lang}).")
        elif foreign_segments:
            critical.append(f"한국어 음성 중 외국어 구간이 {len(foreign_segments)}건 감지되었습니다.")

    # Status decision (priority order from product spec)
    if flags["detectedForeignLanguageTTS"]:
        status = "retry"
    elif len(high_floating) >= 2:
        status = "retry"
    elif flags["detectedIrrelevantObjects"] and irrelevant_findings:
        status = "retry"
    elif flags["detectedDuplicateScenes"] and (sim_score is not None and sim_score < 40):
        status = "retry"
    elif flags["detectedCompanyMixing"]:
        status = "retry"
    elif flags["detectedUnsupportedClaim"]:
        status = "retry"
    elif flags["detectedWrongIndustry"]:
        status = "retry"
    elif high_floating or flags["detectedSpatialDistortion"] or flags["detectedFloatingObjects"]:
        status = "human_review"
    elif flags["detectedVisualTextIssue"]:
        status = "human_review"
    elif critical:
        status = "retry"
    elif len(warnings) >= 3:
        status = "human_review"
    else:
        status = "pass"

    score = max(0, min(100, 100 - len(critical) * 22 - len(warnings) * 4))

    retry_prompt = _build_retry_prompt(client_info, qa_context, critical) if status == "retry" else ""
    human_review_reason = ""
    if status == "human_review":
        bits: list[str] = []
        if flags["detectedFloatingObjects"]:
            bits.append("공중 부유 객체 의심")
        if flags["detectedSpatialDistortion"]:
            bits.append("공간 왜곡 의심")
        if flags["detectedVisualTextIssue"]:
            bits.append("화면 텍스트 문제")
        if not bits and warnings:
            bits = warnings[:3]
        human_review_reason = (
            "사람 검수가 필요합니다: " + "; ".join(bits) if bits else "사람 검수가 필요합니다."
        )

    report = _build_report(legacy_status=status, detailed_findings=detailed_findings)

    return {
        "status": status,
        "score": score,
        "criticalIssues": critical,
        "warnings": warnings,
        **flags,
        **enhanced_flags,
        "sttText": stt_result.get("text") or "",
        "sttLanguage": primary_lang,
        "sceneDiversityScore": sim_score,
        "duplicateSceneRanges": sim_ranges,
        "visualAnomalyFrames": visual_anomaly_frames,
        "irrelevantObjectFindings": irrelevant_findings,
        "audioLanguageSummary": audio_language,
        "visualQaSummary": visual_analysis.get("visualQaSummary") or [],
        "sceneQaSummary": visual_analysis.get("sceneQaSummary") or [],
        "detailedFindings": detailed_findings,
        "report": report,
        "retryPrompt": retry_prompt,
        "humanReviewReason": human_review_reason,
        "checkedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
