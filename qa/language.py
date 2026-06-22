"""Audio language detection from Whisper STT result.

Whisper-1 returns one detected language for the whole audio. For richer
foreign-segment detection we also scan per-segment text for CJK / long Latin
runs while the primary remains Korean.

IMPORTANT: Whisper will report *some* language even for BGM/silence/SFX-only
audio, and that often comes back as Khmer/Thai/Vietnamese on noise. So we first
classify audioType (speech / vocal_music / music / sfx / silence / unclear)
from STT signals (text length, no_speech_prob, avg_logprob). The detected
language is only honored when audioType == "speech" (or vocal_music with
clear lyrics).
"""
from __future__ import annotations

import re
from typing import Any


_CJK_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ]")
_LATIN_RUN_RE = re.compile(r"[A-Za-z]{4,}")
_HANGUL_RE = re.compile(r"[가-힣]")

# Whisper verbose_json returns language as English full name ("korean"), not ISO.
_LANG_MAP = {
    "korean": "ko", "english": "en", "japanese": "ja", "chinese": "zh",
    "spanish": "es", "french": "fr", "german": "de", "russian": "ru",
    "portuguese": "pt", "italian": "it", "thai": "th", "vietnamese": "vi",
    "indonesian": "id", "arabic": "ar", "hindi": "hi", "khmer": "km",
}

_AUDIO_TYPE_LABELS = {
    "speech": "음성",
    "vocal_music": "가창/보컬 음악",
    "music": "BGM",
    "sfx": "효과음",
    "silence": "무음",
    "unclear": "불명확",
}


def _normalize_lang(lang: str | None) -> str:
    l = (lang or "").strip().lower()
    return _LANG_MAP.get(l, l or "unknown")


def _audio_type_label(audio_type: str) -> str:
    return _AUDIO_TYPE_LABELS.get(audio_type, audio_type)


def _avg(values: list[float]) -> float | None:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return sum(vs) / len(vs)


def classify_audio_type(stt_result: dict) -> dict[str, Any]:
    """Classify whether the audio actually contains human speech.

    Returns dict with audioType, speechPresent, reason.

    Heuristics applied to Whisper verbose_json:
    - Empty / near-empty text → silence
    - High average no_speech_prob → music / sfx / unclear
    - Very low avg_logprob with hardly any Hangul → music (Whisper hallucinated
      foreign tokens over BGM)
    - Otherwise → speech
    """
    text = (stt_result.get("text") or "").strip()
    segments = stt_result.get("segments") or []

    avg_no_speech = _avg([s.get("no_speech_prob") for s in segments]) or 0.0
    avg_logprob = _avg([s.get("avg_logprob") for s in segments])
    has_hangul = bool(_HANGUL_RE.search(text))
    has_cjk = bool(_CJK_RE.search(text))
    letters = re.sub(r"[\s\W_]+", "", text)

    if not text or len(letters) < 2:
        return {
            "audioType": "silence",
            "speechPresent": False,
            "reason": "STT 텍스트가 비어 있거나 사실상 비어 있음",
        }

    if avg_no_speech >= 0.6:
        if avg_logprob is not None and avg_logprob <= -1.0 and not has_hangul:
            return {
                "audioType": "music",
                "speechPresent": False,
                "reason": (
                    f"평균 no_speech_prob={avg_no_speech:.2f}, "
                    f"avg_logprob={avg_logprob:.2f} — BGM으로 추정"
                ),
            }
        return {
            "audioType": "unclear",
            "speechPresent": False,
            "reason": f"평균 no_speech_prob={avg_no_speech:.2f} — 음성 여부 불명확",
        }

    if (
        avg_logprob is not None
        and avg_logprob <= -1.2
        and not has_hangul
        and not has_cjk
        and len(letters) < 20
    ):
        return {
            "audioType": "music",
            "speechPresent": False,
            "reason": (
                f"avg_logprob={avg_logprob:.2f}, 의미 있는 음절 거의 없음 — "
                f"비음성(BGM/효과음)으로 추정"
            ),
        }

    if len(letters) < 6 and avg_no_speech >= 0.35:
        return {
            "audioType": "unclear",
            "speechPresent": False,
            "reason": "짧은 텍스트 + 부분적 비음성 신호",
        }

    return {
        "audioType": "speech",
        "speechPresent": True,
        "reason": "STT가 정상적인 음성 텍스트를 반환",
    }


def _build_report(
    *,
    audio_type: str,
    speech_present: bool,
    language_detection_used: bool,
    primary: str,
    foreign_segments: list,
) -> dict[str, str]:
    """Human-readable summary lines shown on the QA report UI."""
    if not speech_present or not language_detection_used:
        return {
            "오디오 유형": _audio_type_label(audio_type),
            "음성 감지": "없음",
            "언어 감지 결과": "무시됨",
            "판단": "외국어 음성 리스크 없음",
        }
    foreign_risk = (primary not in ("ko", "unknown")) or bool(foreign_segments)
    return {
        "오디오 유형": _audio_type_label(audio_type),
        "음성 감지": "있음",
        "언어 감지 결과": primary or "unknown",
        "판단": "외국어 음성 리스크 있음" if foreign_risk else "외국어 음성 리스크 없음",
    }


def detect_audio_language(stt_result: dict) -> dict[str, Any]:
    """Return audioLanguageSummary structure derived from the Whisper response."""
    primary = _normalize_lang(stt_result.get("language"))
    text = (stt_result.get("text") or "")
    segments = stt_result.get("segments") or []

    audio_class = classify_audio_type(stt_result)
    audio_type = audio_class["audioType"]
    speech_present = bool(audio_class["speechPresent"])

    # Non-speech: discard language-detection output entirely. Whisper frequently
    # mis-labels BGM as Khmer / Thai / Vietnamese; we refuse to act on that.
    if not speech_present:
        return {
            "primary": "unknown",
            "confidence": 0.0,
            "detectedSecondary": [],
            "foreignSegments": [],
            "sttMode": stt_result.get("_model") or "whisper-1",
            "audioType": audio_type,
            "speechPresent": False,
            "languageDetectionUsed": False,
            "ignoredDetectedLanguage": primary if primary and primary != "unknown" else None,
            "audioClassificationReason": audio_class["reason"],
            "report": _build_report(
                audio_type=audio_type,
                speech_present=False,
                language_detection_used=False,
                primary=primary,
                foreign_segments=[],
            ),
        }

    foreign_segments: list[dict[str, Any]] = []
    detected_secondary: set[str] = set()

    if primary != "ko":
        # Whole audio detected as non-Korean.
        foreign_segments.append({
            "startSec": 0.0,
            "endSec": None,
            "language": primary,
            "text": text[:240],
        })
        if primary:
            detected_secondary.add(primary)
        confidence = 0.9
    else:
        confidence = 0.95
        for seg in segments:
            t = (seg.get("text") or "").strip()
            if not t:
                continue
            cjk = bool(_CJK_RE.search(t))
            long_latin = bool(_LATIN_RUN_RE.search(t))
            has_korean = bool(_HANGUL_RE.search(t))
            lang: str | None = None
            if cjk and not has_korean:
                lang = "zh-or-ja"
            elif long_latin and not has_korean:
                lang = "en"
            if lang:
                start = float(seg.get("start") or 0.0)
                end = float(seg.get("end") or start)
                foreign_segments.append({
                    "startSec": start,
                    "endSec": end,
                    "language": lang,
                    "text": t[:240],
                })
                detected_secondary.add(lang)

    return {
        "primary": primary,
        "confidence": confidence,
        "detectedSecondary": sorted(detected_secondary),
        "foreignSegments": foreign_segments,
        "sttMode": stt_result.get("_model") or "whisper-1",
        "audioType": "speech",
        "speechPresent": True,
        "languageDetectionUsed": True,
        "audioClassificationReason": audio_class["reason"],
        "report": _build_report(
            audio_type="speech",
            speech_present=True,
            language_detection_used=True,
            primary=primary,
            foreign_segments=foreign_segments,
        ),
    }
