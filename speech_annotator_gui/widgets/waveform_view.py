"""Waveform display widget with colored segment regions and video support."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

_NATIVE_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

_COLOR_AUTO = QColor(100, 150, 255, 100)
_COLOR_VERIFIED = QColor(100, 200, 100, 100)
_COLOR_AUTO_SELECTED = QColor(100, 150, 255, 200)
_COLOR_VERIFIED_SELECTED = QColor(100, 200, 100, 200)


class WaveformView(QWidget):
    """Full-audio waveform with colored segment regions and playback controls."""

    play_all_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._samples: np.ndarray | None = None
        self._sample_rate: float = 0.0
        self._duration: float = 0.0
        self._region_items: list[pg.LinearRegionItem] = []

        layout = QVBoxLayout(self)

        self.plot_widget = pg.PlotWidget(title="Full Audio Waveform")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMaximumHeight(200)
        layout.addWidget(self.plot_widget)

        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play Audio")
        self.btn_stop = QPushButton("Stop")
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        self.btn_play.clicked.connect(self.play_all_requested)
        self.btn_stop.clicked.connect(self.stop_requested)

    def load_audio(self, file_path: Path) -> None:
        """Load an audio file and display its waveform.

        Video files are converted to a temporary WAV via ffmpeg.
        """
        wav_path = file_path
        tmp_wav: Path | None = None

        if file_path.suffix.lower() not in _NATIVE_AUDIO_EXTENSIONS:
            tmp_wav = Path(tempfile.mkstemp(suffix=".wav")[1])
            wav_path = tmp_wav
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(file_path),
                    "-vn", "-ac", "1", "-ar", "16000",
                    "-f", "wav", str(tmp_wav),
                ],
                capture_output=True,
                check=True,
            )

        try:
            data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
            if data.shape[1] > 1:
                data = data.mean(axis=1)
            else:
                data = data[:, 0]
        finally:
            if tmp_wav is not None and tmp_wav.exists():
                tmp_wav.unlink()

        self._samples = data
        self._sample_rate = float(sr)
        self._duration = len(data) / float(sr)

        n_samples = len(data)
        max_points = 5000
        if n_samples > max_points:
            step = n_samples // max_points
            display = data[::step]
        else:
            display = data

        time_axis = np.linspace(0, self._duration, len(display))
        self.plot_widget.plot(
            time_axis, display, clear=True,
            pen=pg.mkPen(color=(0, 100, 200), width=1),
        )

    def get_audio_data(self) -> tuple[np.ndarray, float] | None:
        """Return (samples, sample_rate) if loaded."""
        if self._samples is not None:
            return self._samples, self._sample_rate
        return None

    def set_segments(
        self,
        segments: list[tuple[float, float, bool]],
        selected_index: int,
    ) -> None:
        """Draw colored region items for all segments.

        Args:
            segments: list of (start, end, is_verified) tuples
            selected_index: index of the selected segment (lighter color)
        """
        for item in self._region_items:
            self.plot_widget.removeItem(item)
        self._region_items.clear()

        for i, (start, end, is_verified) in enumerate(segments):
            if i == selected_index:
                color = _COLOR_VERIFIED_SELECTED if is_verified else _COLOR_AUTO_SELECTED
            else:
                color = _COLOR_VERIFIED if is_verified else _COLOR_AUTO

            region = pg.LinearRegionItem(
                values=[start, end],
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(color),
            )
            region.setZValue(10)
            self.plot_widget.addItem(region)
            self._region_items.append(region)
