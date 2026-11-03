"""Tests for speech_annotator_common.srt_reader."""

from __future__ import annotations

from pathlib import Path

from speech_annotator_common.models import Annotation, Segment
from speech_annotator_common.srt_reader import read_srt
from speech_annotator_common.srt_writer import write_srt


def _sample_annotation() -> Annotation:
    return Annotation(
        audio_file="audio/test.wav",
        audio_hash="abc123",
        vad_system="pyannote",
        transcription_system="whisper",
        segments=[
            Segment(start=0.23, end=1.54, text="Hello world."),
            Segment(start=2.1, end=5.67, text="This is a test."),
        ],
        config_yaml="vad_systems:\n- name: pyannote\noutput_format: srt\n",
    )


def test_read_srt_parses_metadata(tmp_path: Path) -> None:
    # Given an SRT written from a known annotation
    annotation = _sample_annotation()
    out = tmp_path / "out.srt"
    write_srt(annotation, out)
    # When reading it back
    parsed = read_srt(out)
    # Then the metadata round-trips
    assert parsed.audio_hash == "abc123"
    assert parsed.vad_system == "pyannote"
    assert parsed.transcription_system == "whisper"
    assert "vad_systems:" in parsed.config_yaml
    assert "output_format: srt" in parsed.config_yaml


def test_read_srt_parses_segments(tmp_path: Path) -> None:
    # Given an SRT with two segments
    annotation = _sample_annotation()
    out = tmp_path / "out.srt"
    write_srt(annotation, out)
    # When reading it back
    parsed = read_srt(out)
    # Then both segments are parsed with correct start/end/text
    assert len(parsed.segments) == 2
    assert parsed.segments[0].start == 0.23
    assert parsed.segments[0].end == 1.54
    assert parsed.segments[0].text == "Hello world."
    assert parsed.segments[1].start == 2.1
    assert parsed.segments[1].end == 5.67
    assert parsed.segments[1].text == "This is a test."


def test_read_srt_handles_multiline_text(tmp_path: Path) -> None:
    # Given an SRT entry with multi-line text
    annotation = Annotation(
        audio_file="audio/test.wav",
        audio_hash="h",
        vad_system="v",
        transcription_system="t",
        segments=[Segment(start=1.0, end=2.0, text="Line one\nLine two")],
        config_yaml="",
    )
    out = tmp_path / "out.srt"
    write_srt(annotation, out)
    # When reading it back
    parsed = read_srt(out)
    # Then the multi-line text is preserved
    assert len(parsed.segments) == 1
    assert parsed.segments[0].text == "Line one\nLine two"
