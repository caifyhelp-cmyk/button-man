"""Thin wrappers around ffmpeg / ffprobe.

- extract_audio: video → 16kHz mono mp3 for Whisper
- capture_frames: representative frames at given offsets + last-2s, full quality
- capture_dense_frames: small frames spaced through the video for hashing only
- frame_to_data_url: inline JPG → data: URL for the web UI
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path


class FfmpegMissing(RuntimeError):
    pass


def _run(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FfmpegMissing(
            f"`{cmd[0]}` not found. Install ffmpeg (see qa/README.md)."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    return proc.stdout


def get_duration_sec(video: Path) -> float:
    out = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", str(video),
    ])
    return float(json.loads(out)["format"]["duration"])


def extract_audio(video: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "libmp3lame", "-b:a", "96k",
        str(out_path),
    ])
    return out_path


def capture_frames(
    video: Path,
    out_dir: Path,
    offsets_sec: list[int | float] | None = None,
) -> list[tuple[Path, float]]:
    """Representative frames for the vision model and the UI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration_sec(video)

    if offsets_sec is None:
        offsets_sec = [0, 2, 5, 10, 15]

    targets: list[float] = []
    for off in offsets_sec:
        if 0 <= off < duration:
            targets.append(float(off))

    end_offset = max(0.0, duration - 2.0)
    if not any(abs(t - end_offset) < 0.25 for t in targets):
        targets.append(end_offset)

    targets = sorted(set(round(t, 2) for t in targets))

    results: list[tuple[Path, float]] = []
    for t in targets:
        out = out_dir / f"frame_{int(t*1000):07d}ms.jpg"
        _run([
            "ffmpeg", "-y", "-ss", f"{t}", "-i", str(video),
            "-frames:v", "1", "-pix_fmt", "yuvj420p", "-q:v", "3", str(out),
        ])
        results.append((out, t))
    return results


def capture_dense_frames(
    video: Path,
    out_dir: Path,
    *,
    target_count: int = 30,
    min_step_sec: float = 0.5,
    max_count: int = 60,
) -> list[tuple[Path, float]]:
    """Small frames sampled across the timeline for perceptual hashing only.

    Step = max(min_step_sec, duration / target_count). Frames are downscaled to
    256px wide and stored at lower quality (cheap to hash, never shown to user).
    Capped at max_count to keep long videos affordable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration_sec(video)
    step = max(min_step_sec, duration / max(1, target_count))

    offsets: list[float] = []
    t = 0.0
    while t < duration and len(offsets) < max_count:
        offsets.append(round(t, 2))
        t += step

    results: list[tuple[Path, float]] = []
    for off in offsets:
        out = out_dir / f"dense_{int(off*1000):07d}ms.jpg"
        _run([
            "ffmpeg", "-y", "-ss", f"{off}", "-i", str(video),
            "-frames:v", "1", "-vf", "scale=256:-1", "-pix_fmt", "yuvj420p", "-q:v", "5",
            str(out),
        ])
        results.append((out, off))
    return results


def frame_to_data_url(path: Path) -> str:
    """Read a JPG file and return a data: URL string for inline embedding."""
    b = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")
