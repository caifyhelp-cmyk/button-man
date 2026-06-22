"""Heuristic mock QA backend — no ffmpeg/STT/LLM.

Same return schema as qa/runner.py + multimodal extensions. Used by /api/qa/run
during MVP so the UI and data flow can be exercised without real analysis.

Mock honesty policy:
- Fields whose signal requires actual frame analysis (floating objects, spatial
  distortion, scene similarity, visual anomalies) default to false/empty, with
  visualQaSummary entries explicitly marked "(mock — 실제 모델 분석 미수행)".
- Fields that *can* be inferred from text inputs (scene-line duplication, foreign
  language in script, industry-incompatible nouns) are heuristically flagged.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


# Per-industry incompatible noun shortlist for the irrelevant-object heuristic.
# Real implementation will be vision-driven; this is a coarse text proxy.
_INDUSTRY_INCOMPATIBLE: dict[str, list[str]] = {
    "동물병원": ["자동차", "비행기", "건설현장", "공장", "주방"],
    "치과": ["자동차", "비행기", "공사장", "주방", "공장"],
    "한의원": ["자동차", "비행기", "주방", "공장"],
    "미용실": ["공장", "건설현장", "트럭", "수술실"],
    "카페": ["수술실", "공장", "건설현장", "트럭"],
    "음식점": ["수술실", "병실", "공사장"],
    "학원": ["수술실", "공장", "술집", "주방"],
    "헬스장": ["수술실", "공장", "주방"],
}


def _find_forbidden(text: str, forbidden_list: list[str]) -> list[str]:
    if not text:
        return []
    return [kw for kw in forbidden_list if kw and kw in text]


def _detect_foreign_in_script(text: str) -> bool:
    if not text:
        return False
    has_korean = bool(re.search(r"[가-힣]", text))
    has_long_latin = bool(re.search(r"[A-Za-z]{4,}", text))
    return has_long_latin and not has_korean


def _detect_brand_mixing(text: str, client_name: str) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    cn_lower = (client_name or "").lower()
    return sorted({t for t in tokens if t.lower() not in cn_lower})[:5]


def _analyze_scenes_text(scenes_text: str) -> dict[str, Any]:
    """Parse the scenes textarea by lines and detect duplicates."""
    if not scenes_text or not scenes_text.strip():
        return {
            "lines": [],
            "uniqueCount": 0,
            "totalCount": 0,
            "diversityScore": None,
            "duplicateGroups": [],
        }
    lines = [l.strip() for l in scenes_text.splitlines() if l.strip()]
    counts: dict[str, list[int]] = {}
    for idx, l in enumerate(lines):
        counts.setdefault(l, []).append(idx)
    duplicate_groups = [
        {"line": k, "indices": v, "count": len(v)}
        for k, v in counts.items() if len(v) > 1
    ]
    unique = len(counts)
    total = len(lines)
    diversity = round((unique / total) * 100, 1) if total else None
    return {
        "lines": lines,
        "uniqueCount": unique,
        "totalCount": total,
        "diversityScore": diversity,
        "duplicateGroups": duplicate_groups,
    }


def _find_irrelevant_objects(industry: str, script: str) -> list[dict[str, Any]]:
    if not industry or not script:
        return []
    incompat = _INDUSTRY_INCOMPATIBLE.get(industry.strip(), [])
    findings: list[dict[str, Any]] = []
    for word in incompat:
        if word in script:
            findings.append({
                "offsetSec": None,
                "object": word,
                "expectedContext": industry,
                "severity": "medium",
                "source": "script-keyword",
            })
    return findings


_AUDIO_TYPE_LABELS = {
    "speech": "음성",
    "vocal_music": "가창/보컬 음악",
    "music": "BGM",
    "sfx": "효과음",
    "silence": "무음",
    "unclear": "불명확",
}


def _classify_mock_audio(script: str | None) -> dict[str, Any]:
    """Mock 모드에서는 실제 오디오 분석이 없으므로 script 길이로 음성 여부를 가정.

    실제 모드에서는 qa.language.classify_audio_type 이 Whisper 신호로 판정한다.
    """
    s = (script or "").strip()
    if not s:
        return {
            "audioType": "music",
            "speechPresent": False,
            "reason": "(mock) 대본/스크립트가 비어 있어 BGM-only 영상으로 가정합니다.",
        }
    if len(re.sub(r"[\s\W_]+", "", s)) < 6:
        return {
            "audioType": "unclear",
            "speechPresent": False,
            "reason": "(mock) 대본이 너무 짧아 사람 음성 여부가 불명확합니다.",
        }
    return {
        "audioType": "speech",
        "speechPresent": True,
        "reason": "(mock) 대본 길이 기준으로 음성으로 가정합니다.",
    }


def _build_audio_summary(script: str, foreign_detected: bool) -> dict[str, Any]:
    cls = _classify_mock_audio(script)
    audio_type = cls["audioType"]
    speech_present = bool(cls["speechPresent"])

    if not speech_present:
        return {
            "primary": "unknown",
            "confidence": 0.0,
            "detectedSecondary": [],
            "foreignSegments": [],
            "sttMode": "mock",
            "audioType": audio_type,
            "speechPresent": False,
            "languageDetectionUsed": False,
            "audioClassificationReason": cls["reason"],
            "report": {
                "오디오 유형": _AUDIO_TYPE_LABELS.get(audio_type, audio_type),
                "음성 감지": "없음",
                "언어 감지 결과": "무시됨",
                "판단": "외국어 음성 리스크 없음",
            },
        }

    primary = "unknown" if foreign_detected else "ko"
    foreign_segs = []
    if foreign_detected:
        foreign_segs = [{
            "startSec": 0.0,
            "endSec": None,
            "language": "unknown",
            "note": "(mock — 대본에 한국어가 거의 없습니다. 실제 STT 필요)",
        }]
    return {
        "primary": primary,
        "confidence": 0.95 if script else 0.5,
        "detectedSecondary": ["ko"] if foreign_detected else [],
        "foreignSegments": foreign_segs,
        "sttMode": "mock",
        "audioType": "speech",
        "speechPresent": True,
        "languageDetectionUsed": True,
        "audioClassificationReason": cls["reason"],
        "report": {
            "오디오 유형": "음성",
            "음성 감지": "있음",
            "언어 감지 결과": primary,
            "판단": "외국어 음성 리스크 있음" if foreign_detected else "외국어 음성 리스크 없음",
        },
    }


def _build_visual_qa_summary(offsets: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "offsetSec": float(o),
            "observations": ["(mock — 실제 비전 모델 분석 미수행)"],
            "spatialOk": None,
            "floatingObjects": [],
            "score": None,
        }
        for o in offsets
    ]


def _build_scene_qa_summary(scene_lines: list[str], industry: str) -> list[dict[str, Any]]:
    if not scene_lines:
        return [{
            "sceneIdx": 0,
            "description": "(scenes 입력 없음)",
            "expectedThemes": [industry] if industry else [],
            "observedThemes": [],
            "matchesIntent": None,
            "note": "씬 정보가 비어 있어 mock 모드에서는 비교할 수 없습니다.",
        }]
    return [
        {
            "sceneIdx": i,
            "description": line,
            "expectedThemes": [industry] if industry else [],
            "observedThemes": [],
            "matchesIntent": None,
            "note": "(mock — 실제 비전 모델로 관찰된 테마 채울 자리)",
        }
        for i, line in enumerate(scene_lines)
    ]


def _build_retry_prompt(client_info: dict, critical: list[str]) -> str:
    name = client_info.get("clientName") or "(고객사명 미입력)"
    industry = client_info.get("industry") or "(업종 미입력)"
    services = client_info.get("services") or []
    promo = client_info.get("promotionPoints") or []
    forbidden = client_info.get("forbiddenClaims") or []
    tone = client_info.get("brandTone") or ""

    lines = [f"[재생성 요청] {name} ({industry})", "", "## 이번 영상에서 수정할 점"]
    if critical:
        lines.extend(f"- {c}" for c in critical)
    else:
        lines.append("- (자동 감지된 치명 문제 없음 — 사람 검수 사유 별도 확인)")
    lines.extend(["", "## 반드시 지킬 것", f"- 고객사명: {name}", f"- 업종: {industry}"])
    if services:
        lines.append(f"- 다룰 서비스: {', '.join(services[:5])}")
    if promo:
        lines.append(f"- 홍보 포인트: {', '.join(promo[:3])}")
    if forbidden:
        lines.append(f"- 금지 표현 (절대 사용 금지): {', '.join(forbidden)}")
    if tone:
        lines.append(f"- 브랜드 톤: {tone}")
    lines.extend([
        "",
        "## 시각 / 장면 원칙",
        "- 공중에 떠 있거나 바닥에 자연스럽게 놓이지 않은 가구·물체가 없어야 합니다.",
        "- 벽·천장·가구 비율의 비정상적 왜곡, 합성 아티팩트가 없어야 합니다.",
        f"- {industry or '해당 업종'}과 무관한 장면(자동차/공장/주방 등)이 등장하지 않아야 합니다.",
        "- 같은 장면이 반복되지 않도록 씬 다양성을 확보해야 합니다.",
        "",
        "## 오디오 원칙",
        "- 한국어 TTS만 사용합니다. 외국어 음성이 섞이지 않아야 합니다.",
        "- client_info에 없는 수치·인증·가격·위치는 단정하지 않습니다.",
    ])
    return "\n".join(lines)


def run_mock(
    *,
    client_info: dict,
    video_meta: dict,
    script: str | None = None,
    scenes: str | None = None,
    generation_prompt: str | None = None,
    references: str | None = None,
) -> dict[str, Any]:
    client_name = (client_info.get("clientName") or "").strip()
    industry = (client_info.get("industry") or "").strip()
    forbidden = client_info.get("forbiddenClaims") or []

    critical: list[str] = []
    warnings: list[str] = []

    foreign_in_script = _detect_foreign_in_script(script or "")
    mock_audio_class = _classify_mock_audio(script or "")
    speech_present_mock = bool(mock_audio_class["speechPresent"])
    # 비음성(=BGM/무음/효과음)이면 외국어 리스크로 처리하지 않는다.
    foreign_lang_tts = foreign_in_script and speech_present_mock
    forbidden_hits = _find_forbidden(script or "", forbidden)
    brand_suspects = _detect_brand_mixing(script or "", client_name)
    irrelevant_findings = _find_irrelevant_objects(industry, script or "")
    scenes_info = _analyze_scenes_text(scenes or "")

    flags = {
        "detectedForeignLanguage": foreign_lang_tts,
        "detectedCompanyMixing": bool(brand_suspects),
        "detectedUnsupportedClaim": bool(forbidden_hits),
        "detectedWrongIndustry": False,
        "detectedVisualTextIssue": False,
        "detectedAudioScriptMismatch": False,
        "detectedFloatingObjects": False,
        "detectedSpatialDistortion": False,
        "detectedIrrelevantObjects": bool(irrelevant_findings),
        "detectedDuplicateScenes": len(scenes_info["duplicateGroups"]) > 0,
        "detectedForeignLanguageTTS": foreign_lang_tts,
    }

    if foreign_lang_tts:
        critical.append("대본에서 한국어가 거의 감지되지 않습니다. 외국어 TTS 가능성이 높습니다.")
    if forbidden_hits:
        critical.append(f"금지 표현이 포함되어 있습니다: {', '.join(forbidden_hits)}")
    if brand_suspects:
        critical.append(f"대본에서 다른 회사명 후보가 발견되었습니다: {', '.join(brand_suspects[:3])}")
    for find in irrelevant_findings:
        critical.append(
            f"업종 '{find['expectedContext']}'과 어울리지 않는 객체 가능성: '{find['object']}'"
        )
    if scenes_info["duplicateGroups"]:
        groups = scenes_info["duplicateGroups"]
        if (scenes_info["diversityScore"] or 100) < 40:
            critical.append(
                f"씬 다양성이 매우 낮습니다 (점수 {scenes_info['diversityScore']}/100, "
                f"중복 그룹 {len(groups)}개)."
            )
        else:
            warnings.append(
                f"중복 씬 라인이 발견됨 (그룹 {len(groups)}개, 다양성 {scenes_info['diversityScore']}/100)."
            )

    if client_name and script and client_name not in script:
        warnings.append(f"대본에 고객사명 '{client_name}'이 등장하지 않습니다.")
    if not industry:
        warnings.append("업종 정보가 비어 있어 업종 일치 여부를 판단할 수 없었습니다.")
    if not client_name:
        warnings.append("고객사명이 비어 있어 비교 기준이 부족합니다.")

    size_bytes = int(video_meta.get("size") or 0)
    size_mb = size_bytes / (1024 * 1024)
    if size_bytes == 0:
        warnings.append("영상 파일 용량이 0입니다.")
    elif size_mb < 0.5:
        warnings.append(f"영상 파일 용량이 매우 작습니다 ({size_mb:.2f} MB).")

    if not script and not scenes and not generation_prompt:
        warnings.append("선택 입력(대본·씬·프롬프트)이 모두 비어 있어 비교 정확도가 낮습니다.")

    warnings.append("(mock) 시각 이상(floating/distortion) 탐지는 실제 비전 모델 연결 후 활성화됩니다.")

    score = max(0, min(100, 100 - len(critical) * 25 - len(warnings) * 4))
    if critical:
        status = "retry"
    elif len(warnings) >= 4:
        status = "human_review"
    else:
        status = "pass"

    retry_prompt = _build_retry_prompt(client_info, critical) if status == "retry" else ""
    human_review_reason = (
        "경고가 다수 발견되어 사람 검수가 필요합니다: " + "; ".join(warnings[:3])
        if status == "human_review" else ""
    )

    if script:
        stt_text = script.strip()
    elif client_name:
        stt_text = f"안녕하세요, {client_name}입니다. (mock STT — 실제 분석 전 단계)"
    else:
        stt_text = "(mock STT — 고객사명·대본이 비어 있어 가짜 텍스트를 사용했습니다)"

    frame_offsets = [0.0, 2.0, 5.0, 10.0, 15.0, -2.0]
    frame_summary = [
        {"offsetSec": o, "note": "(mock — 실제 캡처 미수행)"} for o in frame_offsets
    ]

    visual_qa_summary = _build_visual_qa_summary([o for o in frame_offsets if o >= 0])
    scene_qa_summary = _build_scene_qa_summary(scenes_info["lines"], industry)
    audio_language_summary = _build_audio_summary(script or "", foreign_in_script)

    duplicate_scene_ranges = [
        {
            "sceneLine": g["line"],
            "indices": g["indices"],
            "count": g["count"],
            "note": "텍스트 라인 중복 (실제 프레임 유사도 비교는 mock 미수행)",
        }
        for g in scenes_info["duplicateGroups"]
    ]

    return {
        "status": status,
        "score": score,
        "criticalIssues": critical,
        "warnings": warnings,
        **flags,
        "sttText": stt_text,
        "sttLanguage": audio_language_summary["primary"],
        "frameSummary": frame_summary,
        "retryPrompt": retry_prompt,
        "humanReviewReason": human_review_reason,
        "checkedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),

        "sceneDiversityScore": scenes_info["diversityScore"],
        "duplicateSceneRanges": duplicate_scene_ranges,
        "visualAnomalyFrames": [],
        "irrelevantObjectFindings": irrelevant_findings,
        "audioLanguageSummary": audio_language_summary,
        "visualQaSummary": visual_qa_summary,
        "sceneQaSummary": scene_qa_summary,

        "_meta": {
            "mode": "mock",
            "video": video_meta,
            "inputsProvided": {
                "script": bool(script),
                "scenes": bool(scenes),
                "generationPrompt": bool(generation_prompt),
                "references": bool(references),
            },
            "limitations": [
                "프레임 유사도 비교 미수행 (실제 분석에서 활성화)",
                "공간/구조 왜곡 탐지 미수행",
                "공중 부유 객체 탐지 미수행",
                "실제 STT 미수행 — script를 그대로 사용",
            ],
        },
    }
