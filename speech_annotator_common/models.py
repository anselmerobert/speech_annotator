"""Core data structures for speech annotations."""

from __future__ import annotations

from pydantic import BaseModel


class Segment(BaseModel):
    """A speech segment with start/end times (seconds) and text."""

    start: float
    end: float
    text: str = ""
    confidence: int | None = None


class Annotation(BaseModel):
    """Full annotation for one audio file under one VAD + transcription pair."""

    audio_file: str
    audio_hash: str
    vad_system: str
    transcription_system: str
    segments: list[Segment]
    config_yaml: str
    # GUI verification metadata (populated by the GUI, empty for CLI)
    transcription_systems: str = ""
    verifier_login: str = ""
    gui_name: str = ""
    gui_version: str = ""
    llm_config_yaml: str = ""


def segment_duration(seg: Segment) -> float:
    """Return the duration of a segment in seconds."""
    return seg.end - seg.start
