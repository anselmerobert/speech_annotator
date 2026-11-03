"""Segment-level audio playback widget with color coding."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

_COLOR_AUTO = (100, 150, 255)
_COLOR_VERIFIED = (100, 200, 100)


class SegmentPlayer(QWidget):
    """Plays back a single segment's audio with blue/green color coding."""

    play_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._samples: np.ndarray | None = None
        self._sample_rate: float = 0.0
        self._verified: bool = False

        layout = QVBoxLayout(self)

        self.plot_widget = pg.PlotWidget(title="Segment Audio")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.showGrid(x=True, y=False, alpha=0.3)
        self.plot_widget.setMaximumHeight(100)
        self.plot_widget.hideAxis("left")
        layout.addWidget(self.plot_widget)

        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play Segment")
        self.btn_stop = QPushButton("Stop")
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        self.btn_play.clicked.connect(self.play_requested)
        self.btn_stop.clicked.connect(self.stop_requested)

    def set_verified(self, verified: bool) -> None:
        """Set color coding: blue for auto, green for verified."""
        self._verified = verified
        if self._samples is not None:
            self._redraw()

    def load_segment(
        self,
        full_samples: np.ndarray,
        sample_rate: float,
        start: float,
        end: float,
    ) -> None:
        """Load a segment from the full audio samples."""
        start_idx = max(0, int(start * sample_rate))
        end_idx = min(len(full_samples), int(end * sample_rate))

        self._samples = full_samples[start_idx:end_idx].copy()
        self._sample_rate = sample_rate
        self._redraw()

    def _redraw(self) -> None:
        if self._samples is None:
            return
        color = _COLOR_VERIFIED if self._verified else _COLOR_AUTO
        n = len(self._samples)
        max_points = 2000
        if n > max_points:
            step = n // max_points
            display = self._samples[::step]
        else:
            display = self._samples
        duration = len(self._samples) / self._sample_rate
        time_axis = np.linspace(0, duration, len(display))
        self.plot_widget.plot(
            time_axis, display, clear=True,
            pen=pg.mkPen(color=color, width=1),
        )
