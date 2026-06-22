"""Strip base64 image strings from the QA response before it leaves the server.

The pipeline embeds representative frames as data: URLs inside
frameSummary[].imageDataUrl so the UI can render thumbnails. That payload is
heavy (hundreds of KB per frame × N frames) and we don't want it travelling
through API responses or logs. This module makes a sanitized copy of the
result with image data removed; the original is not mutated, and no other
fields (duplicateSceneRanges, audioLanguageSummary, detected flags,
criticalIssues, _meta) are touched.
"""
from __future__ import annotations

import json
import logging
from typing import Any

_LOG = logging.getLogger(__name__)

_DROP_KEYS = {
    "imageDataUrl",
    "dataUrl",
    "base64",
    "imageBase64",
    "thumbnail",
}

_FRAME_KEEP_KEYS = (
    "index",
    "offsetSec",
    "summary",
    "description",
    "detectedText",
    "visualIssues",
    "observations",
    "note",
)


def _is_data_image(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("data:image/")


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in _DROP_KEYS or _is_data_image(v):
                continue
            out[k] = _strip(v)
        return out
    if isinstance(value, list):
        return [_strip(v) for v in value if not _is_data_image(v)]
    return value


def _slim_frame_summary(frames: list) -> list:
    out: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        slim: dict[str, Any] = {"index": frame.get("index", i)}
        for k in _FRAME_KEEP_KEYS:
            if k == "index":
                continue
            if k in frame and frame[k] is not None and not _is_data_image(frame[k]):
                slim[k] = frame[k]
        out.append(slim)
    return out


def lighten_qa_response(result: dict) -> dict:
    """Return a sanitized copy of the QA result without base64 image data."""
    if not isinstance(result, dict):
        return result
    light = _strip(result)
    if isinstance(result.get("frameSummary"), list):
        light["frameSummary"] = _slim_frame_summary(result["frameSummary"])
    return light


def log_payload_sizes(label: str, original: dict, light: dict) -> None:
    """Print before/after JSON sizes so dev logs show the saving."""
    try:
        before = len(json.dumps(original, ensure_ascii=False))
        after = len(json.dumps(light, ensure_ascii=False))
        delta = before - after
        ratio = (delta / before * 100.0) if before else 0.0
        msg = (
            f"[QA payload] {label} before={before}B after={after}B "
            f"saved={delta}B ({ratio:.1f}%)"
        )
        _LOG.info(msg)
        print(msg)
    except Exception as e:
        print(f"[QA payload] size log failed: {type(e).__name__}: {e}")
