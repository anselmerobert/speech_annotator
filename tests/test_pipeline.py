"""Integration test for the CLI pipeline using mock HTTP servers."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from speech_annotator_cli.pipeline import run_pipeline

# --- Mock HTTP server ---

_VAD_SEGMENTS = [
    {"start": 0.0, "end": 2.5},
    {"start": 3.0, "end": 5.0},
]

_TRANSCRIPTION_TEXT = "Mock transcription text"


class _MockVADHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/vad":
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)  # consume the upload
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"segments": _VAD_SEGMENTS}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args: object) -> None:
        pass  # suppress logs


class _MockTranscriptionHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/transcribe":
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)  # consume the upload
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"text": _TRANSCRIPTION_TEXT, "start": 0.0, "end": 2.5}).encode()
            )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


def _start_server(port: int, handler_cls) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def mock_servers():
    """Start mock VAD and two transcription servers."""
    vad_server = _start_server(18001, _MockVADHandler)
    whisper_server = _start_server(18002, _MockTranscriptionHandler)
    parakeet_server = _start_server(18003, _MockTranscriptionHandler)
    yield
    vad_server.shutdown()
    whisper_server.shutdown()
    parakeet_server.shutdown()


@pytest.fixture
def cli_config(tmp_path: Path) -> Path:
    """Write a CLI config pointing to the mock servers."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "vad_systems:\n"
        "  - name: pyannote\n"
        "    endpoint: http://127.0.0.1:18001\n"
        "transcription_systems:\n"
        "  - name: whisper\n"
        "    endpoint: http://127.0.0.1:18002\n"
        "  - name: parakeet\n"
        "    endpoint: http://127.0.0.1:18003\n"
        "output_format: srt\n",
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def audio_input(tmp_path: Path) -> Path:
    """Create an input dir with a test WAV file."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # Copy the test fixture
    import shutil

    src = Path(__file__).parent / "fixtures" / "audio" / "test_5sec.wav"
    shutil.copy2(src, input_dir / "test.wav")
    return input_dir


def test_cli_pipeline_produces_srt(
    mock_servers, cli_config: Path, audio_input: Path, tmp_path: Path
) -> None:
    """End-to-end: run the CLI pipeline with mock servers and verify SRT output."""
    output_dir = tmp_path / "output"

    asyncio.run(
        run_pipeline(
            config=_load_config(cli_config),
            input_path=audio_input,
            output_path=output_dir,
        )
    )

    # Verify SRT files were created — one per VAD×transcription combo
    srt_files = list(output_dir.rglob("*.srt"))
    assert len(srt_files) == 2  # pyannote_whisper + pyannote_parakeet

    # Check that SRT files are under {vad}_{transcription} subdirectories
    # with the same stem as the input audio and .srt extension
    rel_paths = {f.relative_to(output_dir) for f in srt_files}
    assert Path("pyannote_whisper/test.srt") in rel_paths
    assert Path("pyannote_parakeet/test.srt") in rel_paths

    # Verify SRT content has metadata headers and entries
    for srt_file in srt_files:
        content = srt_file.read_text(encoding="utf-8")
        assert "; audio_hash=" in content
        assert "; vad_system=pyannote" in content
        assert "; transcription_system=" in content
        assert "; config=" in content or "; vad_systems" in content
        # Should have 2 SRT entries (matching 2 VAD segments)
        assert "1\n" in content
        assert "2\n" in content
        assert _TRANSCRIPTION_TEXT in content


def test_cli_pipeline_preserves_directory_structure(
    mock_servers, cli_config: Path, tmp_path: Path
) -> None:
    """Verify that nested input directory structure is preserved in output."""
    import shutil

    input_dir = tmp_path / "input"
    (input_dir / "subdir").mkdir(parents=True)
    src = Path(__file__).parent / "fixtures" / "audio" / "test_5sec.wav"
    shutil.copy2(src, input_dir / "subdir" / "nested.wav")

    output_dir = tmp_path / "output"

    asyncio.run(
        run_pipeline(
            config=_load_config(cli_config),
            input_path=input_dir,
            output_path=output_dir,
        )
    )

    # SRT should be in output/{vad}_{trans}/subdir/
    nested_srts = list(output_dir.rglob("*.srt"))
    assert len(nested_srts) == 2
    rel_paths = {f.relative_to(output_dir) for f in nested_srts}
    assert Path("pyannote_whisper/subdir/nested.srt") in rel_paths
    assert Path("pyannote_parakeet/subdir/nested.srt") in rel_paths


def _load_config(path: Path):
    from speech_annotator_common.config import load_cli_config

    return load_cli_config(path)


def test_cli_pipeline_skips_existing_srt(
    mock_servers, cli_config: Path, audio_input: Path, tmp_path: Path
) -> None:
    """Re-running the pipeline should skip SRT files that already exist with matching metadata."""
    output_dir = tmp_path / "output"

    # First run — produces 2 SRT files
    asyncio.run(
        run_pipeline(
            config=_load_config(cli_config),
            input_path=audio_input,
            output_path=output_dir,
        )
    )
    srt_files = list(output_dir.rglob("*.srt"))
    assert len(srt_files) == 2

    # Record file modification times
    mtimes_before = {f.relative_to(output_dir): f.stat().st_mtime for f in srt_files}

    # Second run — should skip all existing SRTs
    import time
    time.sleep(0.1)  # ensure mtimes would differ if files were rewritten

    asyncio.run(
        run_pipeline(
            config=_load_config(cli_config),
            input_path=audio_input,
            output_path=output_dir,
        )
    )

    srt_files_after = list(output_dir.rglob("*.srt"))
    assert len(srt_files_after) == 2

    # Files should NOT have been rewritten (same mtime)
    for f in srt_files_after:
        rel = f.relative_to(output_dir)
        assert f.stat().st_mtime == mtimes_before[rel], f"{rel} was rewritten"


@pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="ffmpeg not installed"
)
def test_cli_pipeline_extracts_segments_from_vad_wav(
    mock_servers, cli_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Segments must be extracted from the WAV the VAD ran on, not the original media.

    Container timestamps (mp4 & co) can drift from the decoded sample timeline;
    extracting with -ss/-to from the original file then shifts the audio while
    the VAD segments (and the GUI playback) follow the decoded timeline.
    """
    from speech_annotator_cli import pipeline as pipeline_module

    # Given a video file (non-native extension) built from the WAV fixture
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video_path = input_dir / "test.mp4"
    fixture_wav = Path(__file__).parent / "fixtures" / "audio" / "test_5sec.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(fixture_wav),
         "-c:a", "aac", "-y", str(video_path)],
        check=True,
    )

    converted_outputs: list[Path] = []
    real_to_wav = pipeline_module._to_wav

    def _record_to_wav(file_path: Path, output_path: Path) -> None:
        converted_outputs.append(output_path)
        real_to_wav(file_path, output_path)

    extraction_sources: list[Path] = []

    def _record_extract(file_path: Path, start: float, end: float, output_path: Path) -> None:
        extraction_sources.append(file_path)
        shutil.copy2(fixture_wav, output_path)

    monkeypatch.setattr(pipeline_module, "_to_wav", _record_to_wav)
    monkeypatch.setattr(pipeline_module, "extract_segment", _record_extract)

    # When running the pipeline on the video
    asyncio.run(
        run_pipeline(
            config=_load_config(cli_config),
            input_path=input_dir,
            output_path=tmp_path / "output",
        )
    )

    # Then segments were extracted from the converted VAD WAV, never the video
    assert extraction_sources, "extract_segment was never called"
    assert converted_outputs, "no audio conversion happened for the video input"
    assert video_path not in extraction_sources
    assert set(extraction_sources) == set(converted_outputs)
