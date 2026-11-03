"""CLI entry point: python -m speech_annotator_cli --config <yaml> --input <path> --output <dir>."""

from __future__ import annotations

import argparse
from pathlib import Path

from speech_annotator_cli.pipeline import run_pipeline_sync


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="speech_annotator_cli",
        description="Run VAD + transcription over audio/video files and write SRT annotations.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to CLI config YAML")
    parser.add_argument("--input", required=True, type=Path, help="Path to audio/video file or directory")
    parser.add_argument("--output", required=True, type=Path, help="Path to output directory for SRT files")
    args = parser.parse_args()

    run_pipeline_sync(args.config, args.input, args.output)


if __name__ == "__main__":
    main()
