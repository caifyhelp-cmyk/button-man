"""Speech-to-text via OpenAI Whisper. Returns text + detected language."""
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


def transcribe(audio_path: Path) -> dict:
    """Returns {'text': str, 'language': str (ISO-639-1)}.

    Language is NOT forced — we need Whisper to detect non-Korean TTS for QA.
    """
    client = _get_client()
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )
    return {
        "text": getattr(resp, "text", "") or "",
        "language": getattr(resp, "language", "") or "",
    }
