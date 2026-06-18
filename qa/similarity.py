"""Frame similarity (dHash) + duplicate-scene clustering.

Self-contained perceptual hashing — only depends on Pillow. The dHash is robust
to small lighting/encoding changes and gives a 64-bit fingerprint per frame.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def _dhash(img_path: Path, size: int = 8) -> int:
    """Difference hash. Returns 64-bit int (size=8)."""
    with Image.open(img_path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        pixels = list(im.getdata())
    bits = 0
    for row in range(size):
        row_start = row * (size + 1)
        for col in range(size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def calculate_frame_similarity(
    frame_paths: list[tuple[Path, float]],
) -> list[dict]:
    """Hash each frame. Returns [{offsetSec, hash}] in input order."""
    return [{"offsetSec": float(t), "hash": _dhash(p)} for p, t in frame_paths]


def detect_duplicate_scenes(
    hashes: list[dict],
    *,
    similarity_threshold: int = 8,
    min_cluster_size: int = 3,
) -> tuple[list[dict], float | None]:
    """Cluster similar frames → duplicate ranges + diversity score.

    similarity_threshold: max Hamming distance (out of 64) to treat two frames
    as "same shot". 0-8 conservative, 8-14 permissive.
    min_cluster_size: a cluster needs at least N members to count as a
    duplicate range (single matches are noise).
    Returns (duplicateSceneRanges, sceneDiversityScore 0..100).
    """
    if not hashes:
        return [], None

    clusters: list[dict] = []  # {'rep': int, 'members': [{'offsetSec', 'hash'}]}
    for h in hashes:
        placed = False
        for c in clusters:
            if _hamming(h["hash"], c["rep"]) <= similarity_threshold:
                c["members"].append(h)
                placed = True
                break
        if not placed:
            clusters.append({"rep": h["hash"], "members": [h]})

    ranges: list[dict] = []
    for c in clusters:
        if len(c["members"]) < min_cluster_size:
            continue
        offsets = sorted(m["offsetSec"] for m in c["members"])
        ranges.append({
            "startSec": round(offsets[0], 2),
            "endSec": round(offsets[-1], 2),
            "similarFrameOffsets": [round(o, 2) for o in offsets],
            "similarity": round(1 - similarity_threshold / 64, 2),
            "reason": f"{len(offsets)}프레임이 perceptual hash 유사 클러스터에 속함",
        })

    diversity_score = round(len(clusters) / max(1, len(hashes)) * 100, 1)
    return ranges, diversity_score
