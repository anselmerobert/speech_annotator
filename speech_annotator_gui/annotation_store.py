"""Load and index automatic and verified annotations."""

from pathlib import Path

from speech_annotator_common.models import Annotation
from speech_annotator_common.srt_reader import read_srt

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".webm", ".ogg",
    ".avi", ".mp4", ".mkv", ".mov",
}


def scan_audio_files(audio_dir: Path) -> list[Path]:
    """Recursively find all audio/video files in audio_dir."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(audio_dir.rglob(f"*{ext}"))
    return sorted(files)


def load_auto_annotations(annotations_dir: Path) -> dict[str, list[Annotation]]:
    """Load all automatic SRT annotations, indexed by audio file stem.

    Auto annotations are organized as: {annotations_dir}/{vad}_{trans}/{rel_path}/{stem}.srt
    The audio stem is the SRT filename stem (e.g. 'test' from 'test.srt').
    Skips the 'fusion' subdirectory (handled separately).

    Returns mapping: audio_stem → list of Annotation objects.
    """
    result: dict[str, list[Annotation]] = {}
    if not annotations_dir.exists():
        return result

    for srt_file in sorted(annotations_dir.rglob("*.srt")):
        if "fusion" in srt_file.parts:
            continue
        try:
            annotation = read_srt(srt_file)
        except Exception:
            continue

        if annotation.vad_system and annotation.transcription_system:
            stem = srt_file.stem
            result.setdefault(stem, []).append(annotation)

    return result


def load_fusion_annotations(annotations_dir: Path) -> dict[str, Annotation]:
    """Load all fusion SRT annotations, indexed by audio file stem.

    Fusion annotations are in: {annotations_dir}/fusion/{rel_path}/{stem}.srt

    Returns mapping: audio_stem → Annotation object.
    """
    result: dict[str, Annotation] = {}
    fusion_dir = annotations_dir / "fusion"
    if not fusion_dir.exists():
        return result

    for srt_file in sorted(fusion_dir.rglob("*.srt")):
        try:
            annotation = read_srt(srt_file)
        except Exception:
            continue
        stem = srt_file.stem
        result[stem] = annotation

    return result


def load_verified_annotations(output_dir: Path) -> dict[str, Annotation]:
    """Load all verified SRT annotations, indexed by audio file stem.

    Returns mapping: audio_stem → Annotation object.
    """
    result: dict[str, Annotation] = {}
    if not output_dir.exists():
        return result

    for srt_file in sorted(output_dir.rglob("*.srt")):
        name = srt_file.stem
        try:
            annotation = read_srt(srt_file)
        except Exception:
            continue

        # Verified annotations are plain {stem}.srt with no vad/transcription
        if not annotation.vad_system and not annotation.transcription_system:
            result[name] = annotation
        else:
            # It's an auto annotation that ended up in the output dir
            result[name] = annotation

    return result


class AnnotationStore:
    """Indexes automatic and verified annotations for the GUI."""

    def __init__(
        self,
        audio_dir: Path,
        auto_annotations_dir: Path,
        output_dir: Path,
    ) -> None:
        self.audio_dir = audio_dir
        self.auto_annotations_dir = auto_annotations_dir
        self.output_dir = output_dir

        self.audio_files = scan_audio_files(audio_dir)
        self._auto_by_stem = load_auto_annotations(auto_annotations_dir)
        self._fusion_by_stem = load_fusion_annotations(auto_annotations_dir)
        self._verified_by_stem = load_verified_annotations(output_dir)

    def get_audio_files(self) -> list[Path]:
        """Return all audio/video files found."""
        return self.audio_files

    def get_auto_annotations(self, audio_stem: str) -> list[Annotation]:
        """Return all automatic annotations for a given audio file stem."""
        return self._auto_by_stem.get(audio_stem, [])

    def get_verified_annotation(self, audio_stem: str) -> Annotation | None:
        """Return the verified annotation for a given audio file stem, if any."""
        return self._verified_by_stem.get(audio_stem)

    def has_verified(self, audio_stem: str) -> bool:
        """Check if a verified annotation exists for this audio."""
        return audio_stem in self._verified_by_stem

    def get_transcription_count(self, audio_stem: str) -> int:
        """Return the number of automatic transcriptions available for this audio."""
        return len(self._auto_by_stem.get(audio_stem, []))

    def get_fusion_annotation(self, audio_stem: str) -> Annotation | None:
        """Return the fusion annotation for a given audio file stem, if any."""
        return self._fusion_by_stem.get(audio_stem)

    def get_verification_status(self, audio_stem: str) -> str:
        """Return verification status: 'full', 'partial', or 'none'.

        - 'full': verified SRT exists and covers all segments from fusion annotation
        - 'partial': verified SRT exists but covers fewer segments
        - 'none': no verified SRT exists
        """
        verified = self._verified_by_stem.get(audio_stem)
        if verified is None:
            return "none"
        fusion = self._fusion_by_stem.get(audio_stem)
        if fusion is None:
            return "none"
        total_segments = len(fusion.segments)
        if total_segments == 0:
            return "full"
        return "full" if len(verified.segments) >= total_segments else "partial"

    def reload_verified(self) -> None:
        """Reload verified annotations from disk (call after SRT is written)."""
        self._verified_by_stem = load_verified_annotations(self.output_dir)
