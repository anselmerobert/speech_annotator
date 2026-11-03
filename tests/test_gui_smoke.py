"""GUI smoke test using offscreen platform."""

import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import soundfile as sf
import pytest


@pytest.fixture
def test_audio_dir(tmp_path):
    """Create a test audio directory with a short WAV file."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(audio_dir / "test.wav"), samples, sr)
    return audio_dir


@pytest.fixture
def test_llm_config(tmp_path):
    """Create a test LLM config."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "llm_test.yaml"
    config_path.write_text(
        "llm:\n"
        "  base_url: http://localhost:8080/v1\n"
        "  api_key: test-key\n"
        "  model: test-model\n"
    )
    return config_path


def test_gui_launches(test_audio_dir, test_llm_config, tmp_path):
    """Test that the GUI launches without crashing."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)

    from speech_annotator_common.config import load_llm_config
    from speech_annotator_gui.main_window import MainWindow

    config = load_llm_config(test_llm_config)
    output_dir = tmp_path / "output"
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()

    window = MainWindow(
        llm_config=config,
        audio_dir=test_audio_dir,
        auto_annotations_dir=annotations_dir,
        output_dir=output_dir,
    )
    assert window.windowTitle() == "Speech Annotator"
    assert window.file_list.count() == 1  # one audio file
    window.close()


def test_file_list_populates(test_audio_dir, test_llm_config, tmp_path):
    """Test that the file list populates correctly."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)

    from speech_annotator_common.config import load_llm_config
    from speech_annotator_gui.main_window import MainWindow

    config = load_llm_config(test_llm_config)
    output_dir = tmp_path / "output"
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()

    window = MainWindow(
        llm_config=config,
        audio_dir=test_audio_dir,
        auto_annotations_dir=annotations_dir,
        output_dir=output_dir,
    )
    # File list should have one item (test.wav)
    assert window.file_list.count() == 1
    window.close()
