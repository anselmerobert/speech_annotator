"""Regression tests: validated segments must never be dropped from the
verified output SRT, and chart clicks must navigate between segments."""

from __future__ import annotations

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from PySide6.QtWidgets import QApplication

from speech_annotator_common.config import load_llm_config
from speech_annotator_common.models import Annotation, Segment
from speech_annotator_common.srt_reader import read_srt
from speech_annotator_common.srt_writer import write_srt
from speech_annotator_gui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def gui_setup(tmp_path):
    """Audio file + fusion annotation + LLM config + output dir."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sr = 16000
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(audio_dir / "test.wav"), samples, sr)

    auto_dir = tmp_path / "auto_annotations"
    fusion_dir = auto_dir / "fusion"
    fusion_dir.mkdir(parents=True)
    segments = [
        Segment(start=0.0, end=2.0, text="alpha bravo charlie", confidence=50),
        Segment(start=2.5, end=4.0, text="delta echo foxtrot", confidence=80),
        Segment(start=4.5, end=5.0, text="golf hotel", confidence=20),
    ]
    write_srt(
        Annotation(
            audio_file="test.wav", audio_hash="a" * 64,
            vad_system="pyannote", transcription_system="whisper",
            segments=segments, config_yaml="vad_systems:\n- name: pyannote\n",
        ),
        fusion_dir / "test.srt",
    )

    llm_cfg = tmp_path / "llm.yaml"
    llm_cfg.write_text(
        "llm:\n  base_url: http://localhost:8080/v1\n  api_key: k\n  model: m\n")

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return {
        "audio_dir": audio_dir, "auto_dir": auto_dir,
        "output_dir": output_dir, "llm_cfg": llm_cfg,
    }


def _make_window(app, gui_setup, wait_ms=100):
    from PySide6.QtTest import QTest
    window = MainWindow(
        llm_config=load_llm_config(gui_setup["llm_cfg"]),
        auto_annotations_dir=gui_setup["auto_dir"],
        audio_dir=gui_setup["audio_dir"],
        output_dir=gui_setup["output_dir"],
    )
    # Let the auto-select timer fire and load the file.
    QTest.qWait(wait_ms)
    assert len(window._segments_data) == 3
    return window


def _output_path(gui_setup) -> Path:
    return gui_setup["output_dir"] / "test.srt"


def test_update_text_keeps_validated_state():
    """Editing a validated segment must not demote it (data-loss bug)."""
    from speech_annotator_gui.segment_state import SegmentData

    seg = Segment(start=0.0, end=1.0, text="un deux trois", confidence=50)
    sd = SegmentData(index=0, segment=seg, merged_text="un deux trois",
                     confidence=50, auto_texts=[])
    sd.validate()
    assert sd.is_validated

    sd.update_text("un deux trois quatre")
    assert sd.is_validated
    assert sd.word_edit_count == 1


def test_editing_validated_segment_is_persisted(app, gui_setup):
    """Regression: editing an already-validated segment used to drop it
    from the verified SRT on the next save."""
    window = _make_window(app, gui_setup)
    try:
        # Validate the current segment, then edit it afterwards.
        window._validate_segment()
        assert _output_path(gui_setup).exists()
        idx = window._current_order[window._current_pos]
        assert window._segments_data[idx].is_validated

        window.segment_editor.text_edit.setPlainText("alpha bravo corrected text")
        assert window._segments_data[idx].is_validated

        # Close the window — closeEvent must flush the debounced autosave.
        window.close()

        annotation = read_srt(_output_path(gui_setup))
        assert len(annotation.segments) == 1
        assert annotation.segments[0].text == "alpha bravo corrected text"
    finally:
        window.close()

    # A new session must restore the edited text and validated state.
    window2 = _make_window(app, gui_setup)
    try:
        seg2 = window2._segments_data[2]  # "golf hotel", start=4.5
        assert seg2.is_validated
        assert seg2.current_text == "alpha bravo corrected text"
    finally:
        window2.close()


def test_validate_all_remaining_writes_all_segments(app, gui_setup):
    window = _make_window(app, gui_setup)
    try:
        window._validate_all_remaining()
        annotation = read_srt(_output_path(gui_setup))
        assert len(annotation.segments) == 3
        assert window.store.get_verification_status("test") == "full"
    finally:
        window.close()


def test_chart_click_navigates_to_segment(app, gui_setup):
    window = _make_window(app, gui_setup)
    try:
        window._on_chart_segment_clicked(2)
        assert window._current_pos == 2
        assert window.segment_editor.segment_info_label.text().startswith(
            "Segment 3/3")

        # Out-of-bounds clicks are ignored.
        window._on_chart_segment_clicked(99)
        assert window._current_pos == 2
        window._on_chart_segment_clicked(-1)
        assert window._current_pos == 2
    finally:
        window.close()


def test_edit_settle_timer_scheduled_on_edit(app, gui_setup):
    """Text edits debounce chart refresh + save instead of running them
    on every keystroke (reactivity of the text editor)."""
    window = _make_window(app, gui_setup)
    try:
        window._validate_segment()
        window._edit_settle_timer.stop()
        window.segment_editor.text_edit.setPlainText("alpha bravo charlie edit")
        assert window._edit_settle_timer.isActive()
        window._edit_settle_timer.stop()
    finally:
        window.close()


def test_pending_edit_flushed_on_file_switch(app, gui_setup, tmp_path):
    """An edit made just before switching files must be saved to the
    file being left, not lost or written to the new file."""
    # Second audio file + fusion annotation (named so it sorts after
    # "test.wav" and the startup auto-select still picks "test").
    sf.write(str(gui_setup["audio_dir"] / "test2.wav"),
             np.zeros(16000, dtype=np.float32), 16000)
    write_srt(
        Annotation(
            audio_file="test2.wav", audio_hash="b" * 64,
            vad_system="pyannote", transcription_system="whisper",
            segments=[Segment(start=0.0, end=1.0, text="other segment",
                              confidence=50)],
            config_yaml="vad_systems:\n- name: pyannote\n",
        ),
        gui_setup["auto_dir"] / "fusion" / "test2.srt",
    )

    window = _make_window(app, gui_setup)
    try:
        window._validate_segment()
        window.segment_editor.text_edit.setPlainText("alpha bravo charlie edit")
        assert window._edit_settle_timer.isActive()

        # Switch files before the debounce fires.
        window._on_file_selected("test2")
        assert not window._edit_settle_timer.isActive()
        assert window._current_stem == "test2"

        # The edit was flushed to the first file's SRT.
        annotation = read_srt(_output_path(gui_setup))
        assert len(annotation.segments) == 1
        assert annotation.segments[0].text == "alpha bravo charlie edit"
    finally:
        window.close()
