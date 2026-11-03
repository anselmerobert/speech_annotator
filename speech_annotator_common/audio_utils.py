"""Audio utilities: hashing and ffmpeg-based segment extraction."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def compute_hash(file_path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_segment(file_path: Path, start: float, end: float, output_path: Path) -> None:
    """Extract [start, end] seconds from file_path to a 16kHz mono WAV at output_path.

    Uses ffmpeg via subprocess. Raises RuntimeError with stderr content on failure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(file_path),
        "-ss",
        str(start),
        "-to",
        str(end),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        str(output_path),
    ]
    result = subprocess.run(  # noqa: S603 - trusted ffmpeg invocation
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        raise RuntimeError(f"ffmpeg failed (code {result.returncode}): {stderr}")
