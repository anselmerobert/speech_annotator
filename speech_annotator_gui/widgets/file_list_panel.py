"""Left sidebar file list panel with color-coded status indicators."""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QListWidget, QListWidgetItem


class FileListPanel(QListWidget):
    """File list showing audio files with annotation status colors."""

    file_selected = Signal(str)  # emits the audio stem

    COLOR_ALL_VERIFIED = QColor(200, 255, 200)    # green
    COLOR_PARTIAL = QColor(200, 220, 255)         # blue
    COLOR_NONE = QColor(255, 255, 255)            # white (no SRT = default)
    COLOR_NO_ANNOTATIONS = QColor(220, 220, 220)  # gray

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(250)
        self.currentRowChanged.connect(self._on_selection)
        self._stems: dict[int, str] = {}

    def populate(
        self,
        audio_files: list[Path],
        has_fusion,
        get_verification_status,
    ) -> None:
        self.clear()
        self._stems.clear()

        for audio_file in audio_files:
            stem = audio_file.stem
            rel_path = audio_file.name
            item = QListWidgetItem(rel_path)
            row = self.count()
            self._stems[row] = stem

            if not has_fusion(stem):
                item.setBackground(QBrush(self.COLOR_NO_ANNOTATIONS))
            else:
                status = get_verification_status(stem)
                if status == "full":
                    item.setBackground(QBrush(self.COLOR_ALL_VERIFIED))
                elif status == "partial":
                    item.setBackground(QBrush(self.COLOR_PARTIAL))
                else:
                    item.setBackground(QBrush(self.COLOR_NONE))

            self.addItem(item)

    def update_item_color(
        self,
        stem: str,
        has_fusion,
        get_verification_status,
    ) -> None:
        for row, s in self._stems.items():
            if s != stem:
                continue
            item = self.item(row)
            if item is None:
                continue
            if not has_fusion(stem):
                item.setBackground(QBrush(self.COLOR_NO_ANNOTATIONS))
            else:
                status = get_verification_status(stem)
                if status == "full":
                    item.setBackground(QBrush(self.COLOR_ALL_VERIFIED))
                elif status == "partial":
                    item.setBackground(QBrush(self.COLOR_PARTIAL))
                else:
                    item.setBackground(QBrush(self.COLOR_NONE))
            break

    def _on_selection(self, row: int) -> None:
        if row in self._stems:
            self.file_selected.emit(self._stems[row])
