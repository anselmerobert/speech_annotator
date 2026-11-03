"""Main window for the speech annotation GUI."""

import asyncio
import os
from bisect import bisect_left
from pathlib import Path

import sounddevice as sd
import yaml
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

from speech_annotator_common.audio_utils import compute_hash
from speech_annotator_common.config import LlmConfig
from speech_annotator_common.models import Annotation, Segment
from speech_annotator_common.srt_writer import write_srt

from .annotation_store import AnnotationStore
from .llm_merger import propagate_corrections
from .segment_state import SegmentData, SegmentState, _word_edit_distance
from .widgets.word_corrections_chart import WordCorrectionsChart
from .widgets.file_list_panel import FileListPanel
from .widgets.navigation_bar import NavigationBar
from .widgets.segment_editor import SegmentEditor
from .widgets.segment_player import SegmentPlayer
from .widgets.waveform_view import WaveformView

_GUI_NAME = "speech-annotator-gui"
_GUI_VERSION = "0.1.0"


class _PropagateWorker(QObject):
    """Runs LLM correction propagation in a background thread."""

    finished = Signal(list)
    error = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        llm_config: LlmConfig,
        corrections: list[tuple[str, str]],
        texts_to_fix: list[str],
    ) -> None:
        super().__init__()
        self._llm_config = llm_config
        self._corrections = corrections
        self._texts_to_fix = texts_to_fix

    def run(self) -> None:
        try:
            corrected = asyncio.run(propagate_corrections(
                self._llm_config, self._corrections, self._texts_to_fix,
                on_retry=self.status.emit))
            self.finished.emit(corrected)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main annotation GUI window."""

    def __init__(self, llm_config: LlmConfig, audio_dir: Path,
                 auto_annotations_dir: Path, output_dir: Path) -> None:
        super().__init__()
        self.setWindowTitle("Speech Annotator")
        self.resize(1400, 900)
        self.llm_config = llm_config
        self.audio_dir = audio_dir
        self.auto_annotations_dir = auto_annotations_dir
        self.output_dir = output_dir
        self.store = AnnotationStore(audio_dir, auto_annotations_dir, output_dir)
        self._current_stem: str | None = None
        self._segments_data: list[SegmentData] = []
        self._current_order: list[int] = []
        self._current_pos: int = 0
        self._current_auto_annotations: list[Annotation] = []
        self._current_audio_hash: str = ""
        self._propagate_thread: QThread | None = None
        self._propagate_worker: _PropagateWorker | None = None
        self._propagate_indices: list[int] = []
        self._edit_settle_timer = QTimer(self)
        self._edit_settle_timer.setSingleShot(True)
        self._edit_settle_timer.setInterval(400)
        self._edit_settle_timer.timeout.connect(self._on_edit_settled)
        self._build_ui()
        self._populate_file_list()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        self.file_list = FileListPanel()
        self.file_list.file_selected.connect(self._on_file_selected)
        splitter.addWidget(self.file_list)
        workspace = QWidget()
        wl = QVBoxLayout(workspace)

        segment_group = QGroupBox("Current speech segment")
        sg_layout = QVBoxLayout(segment_group)
        self.waveform_view = WaveformView()
        self.waveform_view.play_all_requested.connect(self._play_all_audio)
        self.waveform_view.stop_requested.connect(self._stop_playback)
        sg_layout.addWidget(self.waveform_view)
        self.segment_player = SegmentPlayer()
        self.segment_player.play_requested.connect(self._play_segment)
        self.segment_player.stop_requested.connect(self._stop_playback)
        sg_layout.addWidget(self.segment_player)
        self.segment_editor = SegmentEditor()
        self.segment_editor.text_changed.connect(self._on_text_changed)
        self.segment_editor.validate_requested.connect(self._validate_segment)
        sg_layout.addWidget(self.segment_editor)
        self.navigation_bar = NavigationBar()
        self.navigation_bar.prev_requested.connect(self._go_prev)
        self.navigation_bar.next_requested.connect(self._go_next)
        sg_layout.addWidget(self.navigation_bar)
        wl.addWidget(segment_group)

        self.confidence_chart = WordCorrectionsChart()
        self.confidence_chart.segment_clicked.connect(self._on_chart_segment_clicked)
        wl.addWidget(self.confidence_chart)

        self.btn_validate_all = QPushButton("Validate all remaining")
        self.btn_validate_all.setStyleSheet(
            "QPushButton { background-color: rgb(76,175,80); color: white; "
            "font-weight: bold; font-size: 12px; padding: 6px; }"
        )
        self.btn_validate_all.clicked.connect(self._validate_all_remaining)
        wl.addWidget(self.btn_validate_all)

        self.btn_propagate = QPushButton("Propagate corrections")
        self.btn_propagate.setStyleSheet(
            "QPushButton { background-color: rgb(255,152,0); color: white; "
            "font-weight: bold; font-size: 12px; padding: 6px; }"
        )
        self.btn_propagate.clicked.connect(self._on_propagate_corrections)
        wl.addWidget(self.btn_propagate)

        splitter.addWidget(workspace)
        splitter.setSizes([250, 1150])
        self.setStatusBar(QStatusBar())

    def _populate_file_list(self) -> None:
        files = self.store.get_audio_files()
        self.file_list.populate(
            files,
            has_fusion=lambda stem: self.store.get_fusion_annotation(stem) is not None,
            get_verification_status=self.store.get_verification_status,
        )
        self.statusBar().showMessage(f"Loaded {len(files)} audio files")

        for row in range(self.file_list.count()):
            stem = self.file_list._stems.get(row)
            if stem is None:
                continue
            if self.store.get_fusion_annotation(stem) is None:
                continue
            if self.store.get_verification_status(stem) != "full":
                self.file_list.blockSignals(True)
                self.file_list.setCurrentRow(row)
                self.file_list.blockSignals(False)
                QTimer.singleShot(0, lambda s=stem: self._on_file_selected(s))
                return

    def _on_file_selected(self, stem: str) -> None:
        # Persist any pending edit of the file being left.
        self._flush_pending_edits()
        self._current_stem = stem
        self._segments_data = []
        self._current_order = []
        self._current_pos = 0
        self._current_auto_annotations = []
        self._current_audio_hash = ""
        audio_files = self.store.get_audio_files()
        audio_path = next((f for f in audio_files if f.stem == stem), None)
        if audio_path is None:
            QMessageBox.warning(self, "Error", f"Audio file not found for: {stem}")
            return
        self.waveform_view.load_audio(audio_path)

        fusion = self.store.get_fusion_annotation(stem)
        if fusion is None:
            QMessageBox.critical(
                self, "No Fusion Annotations",
                f"No fusion annotations found for '{stem}'.\n"
                f"Run the CLI with LLM config to generate fusion annotations first.",
            )
            return

        auto_annotations = self.store.get_auto_annotations(stem)
        self._current_auto_annotations = auto_annotations
        self._current_audio_hash = fusion.audio_hash or compute_hash(audio_path)

        for i, seg in enumerate(fusion.segments):
            self._segments_data.append(SegmentData(
                index=i, segment=seg, merged_text=seg.text,
                confidence=seg.confidence if seg.confidence is not None else 50,
                auto_texts=[],
            ))

        verified = self.store.get_verified_annotation(stem)
        if verified:
            self._apply_verified_texts(verified)

        self._compute_order()
        self._show_current_segment()
        self._update_chart()
        n_segments = len(fusion.segments)
        n_validated = sum(1 for s in self._segments_data if s.is_validated)
        self.statusBar().showMessage(
            f"Loaded {n_segments} segments for {stem} "
            f"({n_validated} already validated)"
        )

    def _apply_verified_texts(self, verified: Annotation) -> None:
        """Mark segments matching a verified segment (by start time) as
        validated, restoring their saved text.

        Uses bisect on the sorted verified start times instead of a
        quadratic scan so files with ~1000 segments reload quickly.
        """
        v_sorted = sorted(verified.segments, key=lambda s: s.start)
        v_starts = [s.start for s in v_sorted]
        for sd in self._segments_data:
            start = sd.segment.start
            i = bisect_left(v_starts, start - 0.1)
            if i < len(v_starts) and abs(v_starts[i] - start) < 0.1:
                vseg = v_sorted[i]
                sd.current_text = vseg.text
                sd.state = SegmentState.VALIDATED
                sd.word_edit_count = _word_edit_distance(sd.baseline_text, vseg.text)

    def _on_chart_segment_clicked(self, position: int) -> None:
        """Navigate to the segment clicked in the corrections chart."""
        if not self._current_order:
            return
        if not (0 <= position < len(self._current_order)):
            return
        self._current_pos = position
        self._show_current_segment()
        self.confidence_chart.set_current_position(position)

    def _on_propagate_corrections(self) -> None:
        if self._propagate_thread is not None:
            return

        corrections: list[tuple[str, str]] = []
        for sd in self._segments_data:
            if sd.is_validated and sd.word_edit_count > 0:
                corrections.append((sd.merged_text, sd.current_text))

        if not corrections:
            QMessageBox.information(
                self, "No Corrections",
                "No validated segments with corrections found. "
                "Validate some segments first.",
            )
            return

        unvalidated: list[int] = []
        texts_to_fix: list[str] = []
        for i, sd in enumerate(self._segments_data):
            if not sd.is_validated:
                unvalidated.append(i)
                texts_to_fix.append(sd.current_text)

        if not texts_to_fix:
            QMessageBox.information(
                self, "Nothing to Propagate",
                "All segments are already validated.",
            )
            return

        self._propagate_indices = unvalidated
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        self.statusBar().showMessage("Propagating corrections ...")

        self._propagate_thread = QThread()
        self._propagate_worker = _PropagateWorker(
            self.llm_config, corrections, texts_to_fix)
        self._propagate_worker.moveToThread(self._propagate_thread)
        self._propagate_thread.started.connect(self._propagate_worker.run)
        self._propagate_worker.finished.connect(self._on_propagate_done)
        self._propagate_worker.error.connect(self._on_propagate_error)
        self._propagate_worker.status.connect(
            lambda msg: self.statusBar().showMessage(msg))
        self._propagate_thread.start()

    def _on_propagate_done(self, corrected_texts: list) -> None:
        if self._propagate_thread is not None:
            self._propagate_thread.quit()
            self._propagate_thread.wait()
            self._propagate_thread = None
        self._propagate_worker = None

        for idx, text in zip(self._propagate_indices, corrected_texts, strict=True):
            self._segments_data[idx].apply_propagation(text)

        self._show_current_segment()
        self.confidence_chart.refresh_values()
        n_propagated = len(self._propagate_indices)
        self.statusBar().showMessage(
            f"Propagated corrections to {n_propagated} segments"
        )
        QApplication.restoreOverrideCursor()

    def _on_propagate_error(self, error_msg: str) -> None:
        if self._propagate_thread is not None:
            self._propagate_thread.quit()
            self._propagate_thread.wait()
            self._propagate_thread = None
        self._propagate_worker = None

        QMessageBox.critical(
            self, "Propagation Error",
            f"Failed to propagate corrections: {error_msg}",
        )
        self.statusBar().showMessage("Propagation failed")
        QApplication.restoreOverrideCursor()

    def _compute_order(self) -> None:
        self._current_order = sorted(range(len(self._segments_data)),
                                     key=lambda i: self._segments_data[i].confidence)
        self._current_pos = 0
        for i, idx in enumerate(self._current_order):
            if not self._segments_data[idx].is_validated:
                self._current_pos = i
                return

    def _all_segments_for_waveform(self) -> list[tuple[float, float, bool]]:
        return [(s.segment.start, s.segment.end, s.is_validated)
                for s in self._segments_data]

    def _show_current_segment(self) -> None:
        if not self._current_order:
            return
        idx = self._current_order[self._current_pos]
        sd = self._segments_data[idx]
        self.waveform_view.set_segments(self._all_segments_for_waveform(), idx)
        audio_data = self.waveform_view.get_audio_data()
        if audio_data is not None:
            samples, sr = audio_data
            self.segment_player.load_segment(samples, sr, sd.segment.start, sd.segment.end)
        self.segment_editor.set_segment(
            text=sd.current_text, confidence=sd.confidence,
            total_segments=len(self._segments_data),
            confidence_pos=self._current_pos + 1,
            start=sd.segment.start, end=sd.segment.end)
        self.segment_player.set_verified(sd.is_validated)
        self.segment_editor.set_verified(sd.is_validated)
        self.segment_editor.set_validate_enabled(sd.state != SegmentState.VALIDATED)
        has_corrections = any(
            s.is_validated and s.word_edit_count > 0 for s in self._segments_data
        )
        has_unvalidated = any(not s.is_validated for s in self._segments_data)
        self.btn_propagate.setEnabled(has_corrections and has_unvalidated)
        self.btn_validate_all.setEnabled(has_unvalidated)
        self.navigation_bar.set_at_start(self._current_pos == 0)
        self.navigation_bar.set_at_end(self._current_pos >= len(self._current_order) - 1)

    def _on_text_changed(self, text: str) -> None:
        if self._current_order:
            self._segments_data[self._current_order[self._current_pos]].update_text(text)
            # Debounce chart refresh + save: running them on every
            # keystroke makes the text editor laggy on files with ~1000
            # segments (full bar recompute + chart repaint per keypress).
            self._edit_settle_timer.start()

    def _on_edit_settled(self) -> None:
        """Called once typing pauses: refresh the chart and persist the
        current texts so edits to validated segments are never lost."""
        self.confidence_chart.refresh_values()
        self._save_verified_srt()

    def _flush_pending_edits(self) -> None:
        """Apply a pending debounced refresh/save immediately (used on
        file switch and window close, before the pending edit's target
        file changes)."""
        if self._edit_settle_timer.isActive():
            self._edit_settle_timer.stop()
            self._on_edit_settled()

    def _go_prev(self) -> None:
        if self._current_order and self._current_pos > 0:
            self._current_pos -= 1
            self._show_current_segment()
            self.confidence_chart.set_current_position(self._current_pos)

    def _go_next(self) -> None:
        if self._current_order and self._current_pos < len(self._current_order) - 1:
            self._current_pos += 1
            self._show_current_segment()
            self.confidence_chart.set_current_position(self._current_pos)

    def _validate_all_remaining(self) -> None:
        """Validate all unvalidated segments using their current text."""
        if not self._current_order:
            return
        unvalidated = [
            i for i, s in enumerate(self._segments_data) if not s.is_validated
        ]
        if not unvalidated:
            return
        for i in unvalidated:
            self._segments_data[i].validate()
        self._save_verified_srt()
        self._show_current_segment()
        self.confidence_chart.refresh_values()
        self.store.reload_verified()
        self.file_list.update_item_color(
            self._current_stem,
            has_fusion=lambda stem: self.store.get_fusion_annotation(stem) is not None,
            get_verification_status=self.store.get_verification_status,
        )
        n_validated = sum(1 for s in self._segments_data if s.is_validated)
        n_total = len(self._segments_data)
        self.statusBar().showMessage(
            f"Validated {n_validated}/{n_total} segments"
        )

    def _validate_segment(self) -> None:
        if not self._current_order:
            return
        idx = self._current_order[self._current_pos]
        seg_data = self._segments_data[idx]
        seg_data.current_text = self.segment_editor.get_text()
        seg_data.validate()
        self._save_verified_srt()
        self.waveform_view.set_segments(self._all_segments_for_waveform(), self._current_order[self._current_pos])
        self.segment_player.set_verified(True)
        self.segment_editor.set_verified(True)
        self.segment_editor.set_validate_enabled(False)
        has_corrections = any(
            s.is_validated and s.word_edit_count > 0 for s in self._segments_data
        )
        has_unvalidated = any(not s.is_validated for s in self._segments_data)
        self.btn_propagate.setEnabled(has_corrections and has_unvalidated)
        self.btn_validate_all.setEnabled(has_unvalidated)
        self.confidence_chart.refresh_values()
        self.store.reload_verified()
        self.file_list.update_item_color(
            self._current_stem,
            has_fusion=lambda stem: self.store.get_fusion_annotation(stem) is not None,
            get_verification_status=self.store.get_verification_status,
        )
        n_validated = sum(1 for s in self._segments_data if s.is_validated)
        n_total = len(self._segments_data)
        self.statusBar().showMessage(
            f"Validated {n_validated}/{n_total} segments"
        )

    def _save_verified_srt(self) -> None:
        if not self._current_stem:
            return
        audio_files = self.store.get_audio_files()
        audio_path = next((f for f in audio_files if f.stem == self._current_stem), None)
        if audio_path is None:
            return
        validated_segments = [
            Segment(start=s.segment.start, end=s.segment.end, text=s.current_text)
            for s in self._segments_data if s.is_validated
        ]
        if not validated_segments:
            # Nothing validated yet — do not write an empty verified SRT.
            return

        vad_names = sorted({a.vad_system for a in self._current_auto_annotations if a.vad_system})
        trans_names = sorted(
            {a.transcription_system for a in self._current_auto_annotations if a.transcription_system}
        )
        cli_config_yaml = self._current_auto_annotations[0].config_yaml if self._current_auto_annotations else ""
        llm_config = self.llm_config.model_dump(mode="json")
        llm_config["llm"]["api_key"] = "API_KEY"
        llm_config_yaml = yaml.safe_dump(llm_config, sort_keys=False, allow_unicode=True)

        annotation = Annotation(
            audio_file=self._current_stem,
            audio_hash=self._current_audio_hash,
            vad_system=", ".join(vad_names),
            transcription_system=", ".join(trans_names),
            segments=validated_segments,
            config_yaml=cli_config_yaml,
            transcription_systems=", ".join(trans_names),
            verifier_login=os.getlogin(),
            gui_name=_GUI_NAME,
            gui_version=_GUI_VERSION,
            llm_config_yaml=llm_config_yaml,
        )
        try:
            rel_path = audio_path.relative_to(self.audio_dir)
        except ValueError:
            rel_path = Path(audio_path.name)
        output_path = self.output_dir / rel_path.parent / f"{self._current_stem}.srt"
        write_srt(annotation, output_path)

    def _update_chart(self) -> None:
        if self._segments_data:
            self.confidence_chart.update_chart(
                self._segments_data, self._current_order, self._current_pos)

    def _play_all_audio(self) -> None:
        audio_data = self.waveform_view.get_audio_data()
        if audio_data is not None:
            samples, sr = audio_data
            sd.stop()
            sd.play(samples, sr)

    def _play_segment(self) -> None:
        if not self._current_order:
            return
        seg_data = self._segments_data[self._current_order[self._current_pos]]
        audio_data = self.waveform_view.get_audio_data()
        if audio_data is None:
            return
        samples, sr = audio_data
        si = max(0, int(seg_data.segment.start * sr))
        ei = min(len(samples), int(seg_data.segment.end * sr))
        sd.stop()
        sd.play(samples[si:ei], sr)

    def _stop_playback(self) -> None:
        sd.stop()

    def closeEvent(self, event) -> None:
        """Flush any pending debounced save before closing."""
        self._flush_pending_edits()
        super().closeEvent(event)
