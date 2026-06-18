"""Speech-to-text via OpenAI Whisper. Returns text + language + segments."""
from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def transcribe_audio(audio_path: Path) -> dict:
    """Returns {text, language, segments:[{start,end,text}], _model}.

    Language is NOT forced — we rely on Whisper auto-detection to flag foreign TTS.
    """
    client = _get_client()
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )

    raw_segments = getattr(resp, "segments", None) or []
    segments: list[dict] = []
    for s in raw_segments:
        if isinstance(s, dict):
            segments.append({
                "start": s.get("start"),
                "end": s.get("end"),
                "text": s.get("text"),
            })
        else:
            segments.append({
                "start": getattr(s, "start", None),
                "end": getattr(s, "end", None),
                "text": getattr(s, "text", None),
            })

    return {
        "text": getattr(resp, "text", "") or "",
        "language": getattr(resp, "language", "") or "",
        "segments": segments,
        "_model": "whisper-1",
    }


def transcribe(audio_path: Path) -> dict:
    """Backward-compat alias for callers that don't need segments."""
    return transcribe_audio(audio_path)
