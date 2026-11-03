"""Main orchestration pipeline for the speech annotator CLI."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

import yaml

from speech_annotator_common.audio_utils import compute_hash, extract_segment
from speech_annotator_common.config import CliConfig, LlmConfig, config_to_yaml, load_cli_config
from speech_annotator_common.format_registry import get_writer
from speech_annotator_common.http_clients import (
    MistralOcrClient,
    MistralTranscriptionClient,
    MistralVisionClient,
    TranscriptionClient,
    VadClient,
)
from speech_annotator_common.models import Annotation, Segment
from speech_annotator_common.srt_reader import read_srt
from speech_annotator_common.llm_merger import extract_context_bias_terms, merge_annotations
from speech_annotator_common.srt_writer import write_srt

_AUDIO_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".m4a", ".flac", ".webm", ".ogg", ".avi", ".mp4", ".mkv", ".mov",
})

# Extensions that pyannote/torchaudio can read directly.
_NATIVE_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg"})


def discover_files(input_path: Path) -> list[Path]:
    """Return all audio/video files under input_path (recursive)."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in _AUDIO_EXTENSIONS else []
    return sorted(
        p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS
    )


def _to_wav(file_path: Path, output_path: Path) -> None:
    """Convert any audio/video file to 16kHz mono WAV via ffmpeg."""
    cmd = [
        "ffmpeg", "-i", str(file_path),
        "-ar", "16000", "-ac", "1", "-y", str(output_path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        raise RuntimeError(f"ffmpeg conversion failed: {stderr}")


def _extract_frame(file_path: Path, timestamp: float, output_path: Path) -> bool:
    """Extract a single video frame at the given timestamp (seconds).

    Returns True on success, False on failure.
    """
    cmd = [
        "ffmpeg", "-ss", str(timestamp), "-i", str(file_path),
        "-frames:v", "1", "-q:v", "2", "-y", str(output_path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
    )
    return result.returncode == 0 and output_path.exists()


def _get_video_duration(file_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(file_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        return 0.0
    import json
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0.0))


_VIDEO_EXTENSIONS = frozenset({".avi", ".mp4", ".mkv", ".mov", ".webm"})

# Number of sample frames to send to the vision model for slide detection.
_NUM_DETECTION_FRAMES = 5


async def _detect_and_ocr_slides(
    file_path: Path,
    segments: list[Segment],
    config: CliConfig,
    rel_path: Path,
) -> list[str]:
    """Detect if a video contains presentation slides and, if so, OCR one
    frame per VAD segment to build slide context.

    Returns a list of OCR text per segment (empty strings if not slides or
    if OCR fails).
    """
    ocr_settings = config.mistral_ocr
    if ocr_settings is None or not ocr_settings.api_key:
        return [""] * len(segments)

    if file_path.suffix.lower() not in _VIDEO_EXTENSIONS:
        return [""] * len(segments)

    vision_client = MistralVisionClient(
        ocr_settings.api_key, ocr_settings.vision_model)
    ocr_client = MistralOcrClient(ocr_settings.api_key, ocr_settings.ocr_model)

    # Step 1: Extract sample frames spread across the video for detection.
    duration = _get_video_duration(file_path)
    if duration <= 0:
        return [""] * len(segments)

    detection_timestamps = [
        duration * (i + 1) / (_NUM_DETECTION_FRAMES + 1)
        for i in range(_NUM_DETECTION_FRAMES)
    ]

    detection_frames: list[Path] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for i, ts in enumerate(detection_timestamps):
            frame_path = tmp_root / f"detect_{i}.jpg"
            if _extract_frame(file_path, ts, frame_path):
                detection_frames.append(frame_path)

        if not detection_frames:
            return [""] * len(segments)

        # Step 2: Ask the vision model if these are slides.
        try:
            is_slides = await vision_client.detect_slides(detection_frames)
        except Exception as exc:  # noqa: BLE001
            print(f"[{rel_path}] Slide detection error: {exc}")
            return [""] * len(segments)

        if not is_slides:
            print(f"[{rel_path}] No slides detected, skipping OCR")
            return [""] * len(segments)

        print(f"[{rel_path}] Slides detected, running OCR per segment")

        # Step 3: OCR one frame per VAD segment (midpoint timestamp).
        slide_contexts: list[str] = []
        for idx, seg in enumerate(segments):
            mid_ts = (seg.start + seg.end) / 2
            frame_path = tmp_root / f"seg_{idx}.jpg"
            if not _extract_frame(file_path, mid_ts, frame_path):
                slide_contexts.append("")
                continue
            try:
                ocr_text = await ocr_client.ocr_image(frame_path)
                slide_contexts.append(ocr_text)
                preview = ocr_text[:200].replace("\n", " ")
                if len(ocr_text) > 200:
                    preview += "..."
                print(f"[{rel_path}] OCR segment {idx + 1}/{len(segments)}: "
                      f"{len(ocr_text)} chars")
                print(f"[{rel_path}]   OCR text: {preview}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{rel_path}] OCR segment {idx + 1} error: {exc}")
                slide_contexts.append("")

    return slide_contexts


def _srt_matches(
    srt_path: Path,
    audio_hash: str,
    vad_name: str,
    trans_name: str,
) -> bool:
    """Check if an SRT file exists and its metadata matches the expected annotation."""
    if not srt_path.exists():
        return False
    try:
        existing = read_srt(srt_path)
    except Exception:
        return False
    return (
        existing.audio_hash == audio_hash
        and existing.vad_system == vad_name
        and existing.transcription_system == trans_name
    )


async def _run_fusion(
    file_path: Path, input_root: Path, config: CliConfig, output_path: Path, config_yaml: str,
    slide_contexts: list[str] | None = None,
) -> None:
    audio_hash = compute_hash(file_path)
    rel_path = file_path.relative_to(input_root)
    rel_dir = rel_path.parent
    stem = file_path.stem

    fusion_srt = output_path / "fusion" / rel_dir / f"{stem}.srt"
    if fusion_srt.exists():
        try:
            existing = read_srt(fusion_srt)
            if existing.audio_hash == audio_hash:
                print(f"[{rel_path}] Fusion: already exists, skipping")
                return
        except Exception:
            pass

    first_vad = config.vad_systems[0]
    trans_names = [t.name for t in config.transcription_systems]

    per_system_srts = []
    for trans_name in trans_names:
        srt_path = output_path / f"{first_vad.name}_{trans_name}" / rel_dir / f"{stem}.srt"
        if not srt_path.exists():
            print(f"[{rel_path}] Fusion: missing {srt_path.name}, skipping")
            return
        per_system_srts.append(read_srt(srt_path))

    ref_segments = per_system_srts[0].segments
    n_segments = len(ref_segments)
    tps: list[list[str]] = []
    for si in range(n_segments):
        tps.append([a.segments[si].text for a in per_system_srts if si < len(a.segments)])

    llm_config = LlmConfig(llm=config.llm)
    merged = await merge_annotations(
        llm_config, ref_segments, tps, slide_contexts, label=str(rel_path),
        system_names=trans_names)

    fusion_segments = [
        Segment(start=ref_segments[i].start, end=ref_segments[i].end, text=mtext, confidence=conf)
        for i, (mtext, conf) in enumerate(merged)
    ]

    llm_config_dict = config.llm.model_dump(mode="json")
    llm_config_dict["api_key"] = "API_KEY"
    llm_config_yaml = yaml.safe_dump({"llm": llm_config_dict}, sort_keys=False, allow_unicode=True)

    annotation = Annotation(
        audio_file=str(rel_path),
        audio_hash=audio_hash,
        vad_system=first_vad.name,
        transcription_system=", ".join(trans_names),
        segments=fusion_segments,
        config_yaml=config_yaml,
        transcription_systems=", ".join(trans_names),
        llm_config_yaml=llm_config_yaml,
    )
    write_srt(annotation, fusion_srt)
    print(f"[{rel_path}] Fusion: wrote {n_segments} segments")


async def _process_file(
    file_path: Path, input_root: Path, config: CliConfig, output_path: Path, config_yaml: str
) -> list[str] | None:
    """Process one file: VAD + transcription. Returns slide_contexts (or None)."""
    audio_hash = compute_hash(file_path)
    rel_path = file_path.relative_to(input_root)
    rel_dir = rel_path.parent
    stem = file_path.stem

    # For video files or non-native audio, convert to WAV first for VAD.
    vad_input = file_path
    tmp_wav: Path | None = None
    if file_path.suffix.lower() not in _NATIVE_AUDIO_EXTENSIONS:
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        try:
            _to_wav(file_path, tmp_wav)
            vad_input = tmp_wav
        except Exception as exc:  # noqa: BLE001
            print(f"[{rel_path}] Audio extraction error: {exc}")
            return None

    all_slide_contexts: list[str] | None = None

    try:
        for vad in config.vad_systems:
            # Check if all (vad, trans) SRTs already exist and match
            all_exist = all(
                _srt_matches(
                    output_path / f"{vad.name}_{trans.name}" / rel_dir / f"{stem}.srt",
                    audio_hash, vad.name, trans.name,
                )
                for trans in config.transcription_systems
            )
            if all_exist:
                print(f"[{rel_path}] VAD '{vad.name}': all SRTs already exist, skipping")
                continue

            vad_client = VadClient(vad.name, vad.endpoint)
            try:
                segments = await vad_client.detect_segments(vad_input)
            except Exception as exc:  # noqa: BLE001 - surface and skip this file
                print(f"[{rel_path}] VAD '{vad.name}' error: {exc}")
                return None
            print(f"[{rel_path}] VAD: {len(segments)} segments")

            slide_contexts = await _detect_and_ocr_slides(
                file_path, segments, config, rel_path)
            if all_slide_contexts is None and slide_contexts:
                all_slide_contexts = slide_contexts

            has_ocr = any(ctx for ctx in slide_contexts)
            if has_ocr and config.llm is not None:
                print(f"[{rel_path}] Extracting context_bias terms via LLM")
                llm_config = LlmConfig(llm=config.llm)
                bias_terms: list[list[str]] = []
                for idx, ctx in enumerate(slide_contexts):
                    if ctx:
                        try:
                            terms = await extract_context_bias_terms(llm_config, ctx)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[{rel_path}] context_bias segment {idx + 1} error: {exc}")
                            terms = []
                        bias_terms.append(terms)
                        if terms:
                            print(f"[{rel_path}]   segment {idx + 1} context_bias "
                                  f"({len(terms)} terms): {' '.join(terms[:20])}"
                                  + (" ..." if len(terms) > 20 else ""))
                    else:
                        bias_terms.append([])
                print(f"[{rel_path}] context_bias: "
                      f"{sum(len(t) for t in bias_terms)} terms across "
                      f"{sum(1 for t in bias_terms if t)} segments")
            else:
                bias_terms = [[] for _ in slide_contexts]

            transcriptions: dict[str, list[Segment]] = {}
            for trans in config.transcription_systems:
                srt_path = output_path / f"{vad.name}_{trans.name}" / rel_dir / f"{stem}.srt"
                if _srt_matches(srt_path, audio_hash, vad.name, trans.name):
                    print(f"[{rel_path}] Transcription {trans.name}: SRT already exists, skipping")
                    continue

                if trans.type == "mistral":
                    if not trans.api_key:
                        print(
                            f"[{rel_path}] Transcription '{trans.name}' error: "
                            f"api_key is empty — check that MISTRAL_API_KEY "
                            f"environment variable is set"
                        )
                        return None
                    trans_client = MistralTranscriptionClient(
                        trans.name, trans.api_key, trans.model)
                else:
                    trans_client = TranscriptionClient(trans.name, trans.endpoint)
                seg_copies: list[Segment] = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_root = Path(tmpdir)
                    for idx, seg in enumerate(segments):
                        seg_wav = tmp_root / f"{stem}_{idx}.wav"
                        try:
                            # Extract from the same decoded WAV the VAD ran on:
                            # container timestamps (mp4 & co) can drift from the
                            # decoded sample timeline, shifting -ss/-to cuts.
                            extract_segment(vad_input, seg.start, seg.end, seg_wav)
                            ocr_ctx = slide_contexts[idx] if idx < len(slide_contexts) else ""
                            if trans.type == "mistral":
                                prompt = " ".join(bias_terms[idx]) if idx < len(bias_terms) else ""
                                if prompt:
                                    print(f"[{rel_path}] {trans.name} segment {idx + 1} "
                                          f"context_bias: {prompt[:200]}")
                            else:
                                prompt = ocr_ctx
                                if prompt:
                                    print(f"[{rel_path}] {trans.name} segment {idx + 1} "
                                          f"prompt: {prompt[:200]}")
                            text = await trans_client.transcribe(seg_wav, prompt=prompt)
                        except Exception as exc:  # noqa: BLE001 - skip this file
                            print(f"[{rel_path}] Transcription '{trans.name}' error: {exc}")
                            return None
                        seg_copies.append(Segment(start=seg.start, end=seg.end, text=text))
                        print(f"[{rel_path}] Transcription {trans.name}: "
                              f"{idx + 1}/{len(segments)} done: {text}")
                transcriptions[trans.name] = seg_copies

            # Write one SRT per (vad, transcription) pair.
            for trans in config.transcription_systems:
                if trans.name not in transcriptions:
                    continue
                annotation = Annotation(
                    audio_file=str(rel_path),
                    audio_hash=audio_hash,
                    vad_system=vad.name,
                    transcription_system=trans.name,
                    segments=transcriptions[trans.name],
                    config_yaml=config_yaml,
                )
                out_file = output_path / f"{vad.name}_{trans.name}" / rel_dir / f"{stem}.srt"
                writer = get_writer(config.output_format)
                writer.write(annotation, out_file)
    finally:
        if tmp_wav is not None and tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)
    return all_slide_contexts


async def run_pipeline(config: CliConfig, input_path: Path, output_path: Path) -> None:
    """Discover files, run VAD + transcription, write SRT annotations."""
    files = discover_files(input_path)
    if not files:
        print("No audio/video files found")
        return

    config_yaml = config_to_yaml(config)
    input_root = input_path if input_path.is_dir() else input_path.parent

    for file_path in files:
        slide_contexts = await _process_file(file_path, input_root, config, output_path, config_yaml)
        if config.llm is not None:
            await _run_fusion(
                file_path, input_root, config, output_path, config_yaml,
                slide_contexts)


def run_pipeline_sync(config_path: Path, input_path: Path, output_path: Path) -> None:
    """Synchronous wrapper: load config then run the async pipeline."""
    config = load_cli_config(config_path)
    asyncio.run(run_pipeline(config, input_path, output_path))
