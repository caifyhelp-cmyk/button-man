"""Thin wrappers around ffmpeg / ffprobe."""
from __future__ import annotations

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
    offsets_sec: list[int | float],
) -> list[tuple[Path, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration_sec(video)

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
            "-frames:v", "1", "-q:v", "3", str(out),
        ])
        results.append((out, t))
    return results
