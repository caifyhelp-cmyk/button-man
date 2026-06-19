"""Audio language detection from Whisper STT result.

Whisper-1 returns one detected language for the whole audio. For richer
foreign-segment detection we also scan per-segment text for CJK / long Latin
runs while the primary remains Korean.
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
    "indonesian": "id", "arabic": "ar", "hindi": "hi",
}


def _normalize_lang(lang: str | None) -> str:
    l = (lang or "").strip().lower()
    return _LANG_MAP.get(l, l or "unknown")


def detect_audio_language(stt_result: dict) -> dict[str, Any]:
    """Return audioLanguageSummary structure derived from the Whisper response."""
    primary = _normalize_lang(stt_result.get("language"))
    text = stt_result.get("text") or ""
    segments = stt_result.get("segments") or []

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
    }
