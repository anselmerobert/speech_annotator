"""Format writer protocol and registry for pluggable output formats."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from speech_annotator_common.models import Annotation
from speech_annotator_common.srt_writer import write_srt


class FormatWriter(Protocol):
    def write(self, annotation: Annotation, output_path: Path) -> None: ...


class SrtWriter:
    def write(self, annotation: Annotation, output_path: Path) -> None:
        write_srt(annotation, output_path)


_WRITERS: dict[str, type[FormatWriter]] = {}


def register_writer(name: str, writer_cls: type[FormatWriter]) -> None:
    _WRITERS[name] = writer_cls


def get_writer(name: str) -> FormatWriter:
    try:
        return _WRITERS[name]()
    except KeyError:
        available = ", ".join(sorted(_WRITERS)) or "(none)"
        raise KeyError(f"Unknown format writer '{name}'. Available: {available}") from None


register_writer("srt", SrtWriter)
