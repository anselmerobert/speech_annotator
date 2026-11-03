"""Editable transcription text field for the current segment."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

_BORDER_AUTO = "QFrame { border: 2px solid rgb(100,150,255); border-radius: 4px; }"
_BORDER_VERIFIED = "QFrame { border: 2px solid rgb(100,200,100); border-radius: 4px; }"
_BTN_STYLE = (
    "QPushButton { background-color: rgb(76,175,80); color: white; "
    "font-weight: bold; font-size: 14px; padding: 8px; }"
)


class SegmentEditor(QFrame):
    """Editable transcription text for the current segment."""

    text_changed = Signal(str)
    validate_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(_BORDER_AUTO)

        layout = QVBoxLayout(self)

        self.confidence_label = QLabel("Confidence: —")
        self.confidence_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.confidence_label)

        self.segment_info_label = QLabel("Segment: —")
        layout.addWidget(self.segment_info_label)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Transcription will appear here...")
        self.text_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.text_edit)

        self.btn_validate = QPushButton("✓ Validate")
        self.btn_validate.setStyleSheet(_BTN_STYLE)
        self.btn_validate.clicked.connect(self.validate_requested)
        layout.addWidget(self.btn_validate)

        self._last_text = ""

    def set_segment(
        self,
        text: str,
        confidence: int,
        total_segments: int,
        confidence_pos: int,
        start: float,
        end: float,
    ) -> None:
        self.confidence_label.setText(f"Confidence: {confidence}/100")
        self.segment_info_label.setText(
            f"Segment {confidence_pos}/{total_segments} "
            f"({start:.2f}s - {end:.2f}s)"
        )
        self._last_text = text
        self.text_edit.setPlainText(text)

    def set_verified(self, verified: bool) -> None:
        self.setStyleSheet(_BORDER_VERIFIED if verified else _BORDER_AUTO)

    def set_validate_enabled(self, enabled: bool) -> None:
        self.btn_validate.setEnabled(enabled)

    def get_text(self) -> str:
        return self.text_edit.toPlainText()

    def _on_text_changed(self) -> None:
        new_text = self.text_edit.toPlainText()
        if new_text != self._last_text:
            self._last_text = new_text
            self.text_changed.emit(new_text)

    def clear_editor(self) -> None:
        self.confidence_label.setText("Confidence: —")
        self.segment_info_label.setText("Segment: —")
        self._last_text = ""
        self.text_edit.clear()
