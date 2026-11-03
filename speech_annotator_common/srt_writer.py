"""Write SRT files with comment headers carrying annotation metadata."""

from __future__ import annotations

from pathlib import Path

from speech_annotator_common.models import Annotation


def format_timestamp(seconds: float) -> str:
    """Format seconds as 'HH:MM:SS,mmm' (SRT timestamp, comma decimal)."""
    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(annotation: Annotation, output_path: Path) -> None:
    """Write an Annotation to an SRT file with metadata comment headers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"; audio_hash={annotation.audio_hash}")
    lines.append(f"; vad_system={annotation.vad_system}")
    lines.append(f"; transcription_system={annotation.transcription_system}")
    if annotation.transcription_systems:
        lines.append(f"; transcription_systems={annotation.transcription_systems}")
    if annotation.verifier_login:
        lines.append(f"; verifier={annotation.verifier_login}")
    if annotation.gui_name:
        lines.append(f"; gui_name={annotation.gui_name}")
    if annotation.gui_version:
        lines.append(f"; gui_version={annotation.gui_version}")
    for cfg_line in annotation.config_yaml.splitlines():
        lines.append(f"; {cfg_line}")
    if annotation.llm_config_yaml:
        lines.append("; --- LLM config ---")
        for cfg_line in annotation.llm_config_yaml.splitlines():
            lines.append(f"; {cfg_line}")
    lines.append("")

    for idx, seg in enumerate(annotation.segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
        if seg.confidence is not None:
            lines.append(f"; confidence={seg.confidence}")
        lines.append(seg.text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
