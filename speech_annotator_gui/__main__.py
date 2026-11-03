"""GUI entry point for the speech annotation tool."""

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from speech_annotator_common.config import load_llm_config

from .main_window import MainWindow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speech Annotator GUI — manual verification of annotations"
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to LLM config YAML file",
    )
    parser.add_argument(
        "--audio-dir", required=True, type=Path,
        help="Directory containing audio/video files",
    )
    parser.add_argument(
        "--auto-annotations", required=True, type=Path,
        help="Directory containing automatic annotations (SRT files)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output directory for verified annotations",
    )

    args = parser.parse_args()

    config = load_llm_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    window = MainWindow(
        llm_config=config,
        audio_dir=args.audio_dir,
        auto_annotations_dir=args.auto_annotations,
        output_dir=args.output,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
