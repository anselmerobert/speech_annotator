"""Tests for the remux_mp4 CLI script."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import remux_mp4

ffmpeg_available = shutil.which("ffmpeg") is not None


def _make_mp4(path: Path, duration: int = 10) -> None:
    """Create a healthy MP4 (testsrc video + sine audio)."""
    subprocess.run(
        ["ffmpeg", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=15",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "64k",
         "-y", str(path)],
        check=True,
    )


def _stretch(src: Path, dst: Path, factor: float) -> None:
    """Create a defective copy: audio timeline stretched via -itsscale."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-itsscale", str(factor), "-i", str(src),
         "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-y", str(dst)],
        check=True,
    )


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not installed")
def test_healthy_mp4_not_flagged(tmp_path: Path) -> None:
    # Given a freshly muxed MP4 with consistent timestamps
    healthy = tmp_path / "healthy.mp4"
    _make_mp4(healthy)
    # When analyzing it
    status, info = remux_mp4.analyze(healthy)
    # Then it is reported healthy
    assert status == "ok"
    assert abs(info["gap"]) < 0.25


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not installed")
def test_stretched_mp4_detected(tmp_path: Path) -> None:
    # Given an MP4 whose audio timeline was stretched by 5%
    healthy = tmp_path / "orig.mp4"
    _make_mp4(healthy)
    stretched = tmp_path / "stretched.mp4"
    _stretch(healthy, stretched, 1.05)
    # When analyzing it
    status, info = remux_mp4.analyze(stretched)
    # Then the stretch is detected with the expected gap
    assert status == "stretched"
    assert info["gap"] > 0.25
    assert info["has_video"] is True


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not installed")
def test_stretched_mp4_remuxed_by_main(tmp_path: Path) -> None:
    # Given a directory containing a healthy and a stretched MP4
    healthy = tmp_path / "orig.mp4"
    _make_mp4(healthy)
    stretched = tmp_path / "stretched.mp4"
    _stretch(healthy, stretched, 1.05)
    # When running the CLI on the directory
    exit_code = remux_mp4.main([str(tmp_path)])
    # Then it succeeds and only the stretched file got a _remuxed sibling
    assert exit_code == 0
    fixed = tmp_path / "stretched_remuxed.mp4"
    assert fixed.exists()
    assert not (tmp_path / "orig_remuxed.mp4").exists()
    # And the fixed file is healthy and close to the original duration
    status, info = remux_mp4.analyze(fixed)
    assert status == "ok"
    orig_status, orig_info = remux_mp4.analyze(healthy)
    assert abs(info["declared"] - orig_info["declared"]) < 0.5


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not installed")
def test_dry_run_generates_nothing(tmp_path: Path) -> None:
    # Given a directory containing a stretched MP4
    healthy = tmp_path / "orig.mp4"
    _make_mp4(healthy)
    _stretch(healthy, tmp_path / "stretched.mp4", 1.05)
    # When running the CLI with --dry-run
    exit_code = remux_mp4.main([str(tmp_path), "--dry-run"])
    # Then no _remuxed file is written
    assert exit_code == 0
    assert not (tmp_path / "stretched_remuxed.mp4").exists()
