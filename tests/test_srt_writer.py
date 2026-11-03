"""Tests for speech_annotator_common.srt_writer."""

from __future__ import annotations

from pathlib import Path

from speech_annotator_common.models import Annotation, Segment
from speech_annotator_common.srt_writer import write_srt


def _sample_annotation(audio_file: str = "audio/test.wav") -> Annotation:
    return Annotation(
        audio_file=audio_file,
        audio_hash="abc123",
        vad_system="pyannote",
        transcription_system="whisper",
        segments=[
            Segment(start=0.23, end=1.54, text="Hello world."),
            Segment(start=2.1, end=5.67, text="This is a test."),
        ],
        config_yaml="vad_systems:\n- name: pyannote\noutput_format: srt\n",
    )


def test_write_srt_creates_file(tmp_path: Path) -> None:
    # Given an annotation and an output path
    annotation = _sample_annotation()
    out = tmp_path / "out.srt"
    # When writing the SRT
    write_srt(annotation, out)
    # Then the file exists
    assert out.exists()
    assert out.stat().st_size > 0


def test_srt_has_metadata_headers(tmp_path: Path) -> None:
    # Given a written SRT
    annotation = _sample_annotation()
    out = tmp_path / "out.srt"
    write_srt(annotation, out)
    content = out.read_text(encoding="utf-8")
    # Then the metadata comment headers are present
    assert "; audio_hash=abc123" in content
    assert "; vad_system=pyannote" in content
    assert "; transcription_system=whisper" in content
    assert "; vad_systems:" in content
    assert "; output_format: srt" in content


def test_srt_has_correct_entries(tmp_path: Path) -> None:
    # Given a written SRT with two segments
    annotation = _sample_annotation()
    out = tmp_path / "out.srt"
    write_srt(annotation, out)
    content = out.read_text(encoding="utf-8")
    # Then both entries appear with correct timestamps and text
    assert "1\n00:00:00,230 --> 00:00:01,540\nHello world." in content
    assert "2\n00:00:02,100 --> 00:00:05,670\nThis is a test." in content
    # And there are exactly two numbered entries
    numbered_lines = [ln for ln in content.splitlines() if ln.strip() in {"1", "2"}]
    assert len(numbered_lines) == 2


def test_srt_preserves_directory_structure(tmp_path: Path) -> None:
    # Given an output path in a nested directory that does not exist yet
    annotation = _sample_annotation(audio_file="sub/dir/test.wav")
    out = tmp_path / "output" / "sub" / "dir" / "test_pyannote_whisper.srt"
    # When writing the SRT
    write_srt(annotation, out)
    # Then the nested parent directories were created and the file exists
    assert out.exists()
    assert out.parent.is_dir()
