"""Navigation bar with Previous/Next buttons (confidence ascending only)."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton


class NavigationBar(QFrame):
    """Navigation controls for moving between segments by confidence order."""

    prev_requested = Signal()
    next_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QHBoxLayout(self)
        layout.addStretch()

        self.btn_prev = QPushButton("◀ Previous segment")
        self.btn_prev.setFixedHeight(40)
        layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next segment ▶")
        self.btn_next.setFixedHeight(40)
        layout.addWidget(self.btn_next)

        layout.addStretch()

        self.btn_prev.clicked.connect(self.prev_requested)
        self.btn_next.clicked.connect(self.next_requested)

    def set_at_start(self, at_start: bool) -> None:
        self.btn_prev.setEnabled(not at_start)

    def set_at_end(self, at_end: bool) -> None:
        self.btn_next.setEnabled(not at_end)
