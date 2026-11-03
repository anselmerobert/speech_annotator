"""Parse SRT files (with comment headers) back into Annotation objects."""

from __future__ import annotations

from pathlib import Path

from speech_annotator_common.models import Annotation, Segment

_META_PREFIX = "; "
_CONFIG_PREFIX = "; config="


def _parse_timestamp(ts: str) -> float:
    """Parse 'HH:MM:SS,mmm' into seconds (float)."""
    time_part, ms_part = ts.split(",")
    hours, minutes, secs = time_part.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(ms_part) / 1000.0


def read_srt(file_path: Path) -> Annotation:
    """Read an SRT file with comment headers and return an Annotation."""
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    audio_hash = ""
    vad_system = ""
    transcription_system = ""
    transcription_systems = ""
    verifier_login = ""
    gui_name = ""
    gui_version = ""
    config_lines: list[str] = []
    llm_config_lines: list[str] = []

    in_llm_config = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(_META_PREFIX):
            break
        content = line[len(_META_PREFIX):]
        if content.startswith("audio_hash="):
            audio_hash = content[len("audio_hash="):]
        elif content.startswith("vad_system="):
            vad_system = content[len("vad_system="):]
        elif content.startswith("transcription_system="):
            transcription_system = content[len("transcription_system="):]
        elif content.startswith("transcription_systems="):
            transcription_systems = content[len("transcription_systems="):]
        elif content.startswith("verifier="):
            verifier_login = content[len("verifier="):]
        elif content.startswith("gui_name="):
            gui_name = content[len("gui_name="):]
        elif content.startswith("gui_version="):
            gui_version = content[len("gui_version="):]
        elif content == "--- LLM config ---":
            in_llm_config = True
        elif in_llm_config:
            llm_config_lines.append(content)
        else:
            config_lines.append(content)
        i += 1

    config_yaml = "\n".join(config_lines)
    llm_config_yaml = "\n".join(llm_config_lines)

    segments: list[Segment] = []
    remaining = lines[i:]
    j = 0
    while j < len(remaining):
        line = remaining[j].strip()
        if line == "":
            j += 1
            continue
        if not line.isdigit():
            j += 1
            continue
        j += 1
        if j >= len(remaining):
            break
        time_line = remaining[j].strip()
        j += 1
        confidence: int | None = None
        text_lines: list[str] = []
        while j < len(remaining) and remaining[j].strip() != "":
            line_content = remaining[j].strip()
            if line_content.startswith("; confidence="):
                try:
                    confidence = int(line_content[len("; confidence="):])
                except ValueError:
                    pass
            else:
                text_lines.append(remaining[j])
            j += 1
        start_str, end_str = time_line.split(" --> ")
        segments.append(
            Segment(
                start=_parse_timestamp(start_str),
                end=_parse_timestamp(end_str),
                text="\n".join(text_lines),
                confidence=confidence,
            )
        )

    return Annotation(
        audio_file="",
        audio_hash=audio_hash,
        vad_system=vad_system,
        transcription_system=transcription_system,
        segments=segments,
        config_yaml=config_yaml,
        transcription_systems=transcription_systems,
        verifier_login=verifier_login,
        gui_name=gui_name,
        gui_version=gui_version,
        llm_config_yaml=llm_config_yaml,
    )
