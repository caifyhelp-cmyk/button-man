"""Heuristic mock QA backend — no ffmpeg/STT/LLM.

Same return schema as qa/runner.py. Used by the web route /api/qa/run during MVP
so the UI and data flow can be exercised end-to-end before real analysis is wired in.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


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


def _build_retry_prompt(client_info: dict, critical: list[str]) -> str:
    name = client_info.get("clientName") or "(고객사명 미입력)"
    industry = client_info.get("industry") or "(업종 미입력)"
    services = client_info.get("services") or []
    promo = client_info.get("promotionPoints") or []
    forbidden = client_info.get("forbiddenClaims") or []
    tone = client_info.get("brandTone") or ""

    lines = [f"[재생성 요청] {name} ({industry})", "", "## 이번 영상에서 수정할 점"]
    lines.extend(f"- {c}" for c in critical) if critical else lines.append("- (자동 감지된 문제 없음)")
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
        "## 원칙",
        "- 다른 회사명, 다른 업종 장면, 외국어 TTS는 사용하지 않습니다.",
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
    flags = {
        "detectedForeignLanguage": False,
        "detectedCompanyMixing": False,
        "detectedUnsupportedClaim": False,
        "detectedWrongIndustry": False,
        "detectedVisualTextIssue": False,
        "detectedAudioScriptMismatch": False,
    }

    if script:
        stt_text = script.strip()
    elif client_name:
        stt_text = f"안녕하세요, {client_name}입니다. (mock STT — 실제 분석 전 단계)"
    else:
        stt_text = "(mock STT — 고객사명·대본이 비어 있어 가짜 텍스트를 사용했습니다)"

    if _detect_foreign_in_script(script or ""):
        flags["detectedForeignLanguage"] = True
        critical.append("대본에서 한국어가 거의 감지되지 않습니다. 외국어 TTS 가능성이 있습니다.")

    hits = _find_forbidden(script or "", forbidden)
    if hits:
        flags["detectedUnsupportedClaim"] = True
        critical.append(f"금지 표현이 포함되어 있습니다: {', '.join(hits)}")

    brand_suspects = _detect_brand_mixing(script or "", client_name)
    if brand_suspects:
        flags["detectedCompanyMixing"] = True
        critical.append(
            f"대본에서 다른 회사명 후보가 발견되었습니다: {', '.join(brand_suspects[:3])}"
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

    score = max(0, min(100, 100 - len(critical) * 25 - len(warnings) * 5))
    if critical:
        status = "retry"
    elif len(warnings) >= 3:
        status = "human_review"
    else:
        status = "pass"

    retry_prompt = _build_retry_prompt(client_info, critical) if status == "retry" else ""
    human_review_reason = (
        "경고가 다수 발견되어 사람 검수가 필요합니다: " + "; ".join(warnings[:3])
        if status == "human_review" else ""
    )

    frame_summary = [
        {"offsetSec": float(o), "note": "(mock — 실제 캡처 미수행)"}
        for o in (0, 2, 5, 10, 15)
    ]
    frame_summary.append({"offsetSec": -2.0, "note": "(mock end-2s — 실제 길이 모름)"})

    return {
        "status": status,
        "score": score,
        "criticalIssues": critical,
        "warnings": warnings,
        **flags,
        "sttText": stt_text,
        "sttLanguage": "unknown" if flags["detectedForeignLanguage"] else "ko",
        "frameSummary": frame_summary,
        "retryPrompt": retry_prompt,
        "humanReviewReason": human_review_reason,
        "checkedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_meta": {
            "mode": "mock",
            "video": video_meta,
            "inputsProvided": {
                "script": bool(script),
                "scenes": bool(scenes),
                "generationPrompt": bool(generation_prompt),
                "references": bool(references),
            },
        },
    }
