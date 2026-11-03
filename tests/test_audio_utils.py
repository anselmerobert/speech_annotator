"""Tests for speech_annotator_common.audio_utils."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from speech_annotator_common.audio_utils import compute_hash, extract_segment

_FIXTURE_AUDIO = Path(__file__).parent / "fixtures" / "audio" / "test_5sec.wav"

ffmpeg_available = shutil.which("ffmpeg") is not None


def test_compute_hash_same_file_same_hash(tmp_path: Path) -> None:
    # Given a file written twice with identical content
    content = b"hello world\n" * 100
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(content)
    f2.write_bytes(content)
    # When hashing both
    h1 = compute_hash(f1)
    h2 = compute_hash(f2)
    # Then the hashes are equal
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex length


def test_compute_hash_different_content_different_hash(tmp_path: Path) -> None:
    # Given two files with different content
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"content one")
    f2.write_bytes(b"content two")
    # When hashing both
    h1 = compute_hash(f1)
    h2 = compute_hash(f2)
    # Then the hashes differ
    assert h1 != h2


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not installed")
def test_extract_segment_creates_nonempty_wav(tmp_path: Path) -> None:
    # Given the 5-second sine WAV fixture
    assert _FIXTURE_AUDIO.exists()
    out = tmp_path / "seg.wav"
    # When extracting seconds [1.0, 3.0]
    extract_segment(_FIXTURE_AUDIO, 1.0, 3.0, out)
    # Then the output file exists and is non-empty
    assert out.exists()
    assert out.stat().st_size > 0
    # And it is a valid RIFF/WAV file
    with out.open("rb") as f:
        header = f.read(4)
    assert header == b"RIFF"
