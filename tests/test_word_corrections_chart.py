"""Tests for WordCorrectionsChart: arrows, click navigation, refresh."""

from __future__ import annotations

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from speech_annotator_common.models import Segment
from speech_annotator_gui.segment_state import SegmentData
from speech_annotator_gui.widgets.word_corrections_chart import WordCorrectionsChart


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


def _wait(app, ms=100):
    from PySide6.QtTest import QTest
    QTest.qWait(ms)


def _make_segments() -> list[SegmentData]:
    segs = []
    for i in range(4):
        seg = Segment(
            start=float(i * 10), end=i * 10 + 5.0,
            text=f"baseline text number {i}", confidence=(i + 1) * 20,
        )
        segs.append(SegmentData(
            index=i, segment=seg, merged_text=f"baseline text number {i}",
            confidence=(i + 1) * 20, auto_texts=[],
        ))
    return segs


def _prepared_chart(app):
    """Chart with 4 segments (index 1 validated), order [2, 0, 3, 1]."""
    chart = WordCorrectionsChart()
    chart.resize(600, 400)
    chart.show()
    _wait(app)
    segs = _make_segments()
    segs[1].validate()
    order = [2, 0, 3, 1]
    chart.update_chart(segs, order, current_pos=0)
    _wait(app)
    return chart, segs, order


def test_arrows_mark_unvalidated_segments(app):
    chart, segs, order = _prepared_chart(app)
    try:
        # order = [2, 0, 3, 1]: positions 0, 1, 2 are unvalidated
        arrows = chart.chart_view._arrows
        assert len(arrows) == 3
        arrow_x = [a[0] for a in arrows]
        plot = chart.chart.plotArea()
        slot = plot.width() / 4
        for position in (0, 1, 2):
            expected_x = plot.x() + (position + 0.5) * slot
            assert any(abs(x - expected_x) < 1.0 for x in arrow_x)
        # Arrows must sit above the sentinel bar top, inside the plot area
        for _, bar_top in arrows:
            assert bar_top > plot.y()
    finally:
        chart.close()


def test_click_on_bar_emits_segment_position(app):
    chart, segs, order = _prepared_chart(app)
    try:
        received: list[int] = []
        chart.segment_clicked.connect(received.append)

        plot = chart.chart.plotArea()
        slot = plot.width() / 4
        position = 2
        x = plot.x() + (position + 0.5) * slot
        y = plot.y() + plot.height() / 2
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress, QPointF(x, y), QPointF(x, y),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        chart.chart_view.mousePressEvent(event)
        assert received == [position]
    finally:
        chart.close()


def test_click_outside_plot_area_is_ignored(app):
    chart, segs, order = _prepared_chart(app)
    try:
        received: list[int] = []
        chart.segment_clicked.connect(received.append)

        event = QMouseEvent(
            QEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        chart.chart_view.mousePressEvent(event)
        assert received == []
    finally:
        chart.close()


def test_refresh_values_updates_validated_bar(app):
    chart, segs, order = _prepared_chart(app)
    try:
        # seg index 1 is at position 3 in the order
        segs[1].update_text("baseline text number 1 now with corrections")
        assert segs[1].is_validated  # editing must not demote a validated segment
        chart.refresh_values()
        _wait(app, 50)
        # The validated bar (position 3) should now show corrections > 0
        assert chart._bar_validated.at(3) > 0
        # Still exactly 3 arrows (validation states unchanged)
        assert len(chart.chart_view._arrows) == 3
    finally:
        chart.close()


def test_set_current_position_moves_highlight(app):
    chart, segs, order = _prepared_chart(app)
    try:
        assert chart.chart_view._rect is not None
        rect_before = chart.chart_view._rect
        chart.set_current_position(3)
        _wait(app, 50)
        rect_after = chart.chart_view._rect
        assert rect_after is not None
        assert rect_after.x() > rect_before.x()
        # Out-of-range positions are ignored
        chart.set_current_position(99)
        _wait(app, 50)
        assert chart.chart_view._rect is not None
    finally:
        chart.close()


def test_title_shows_validation_progress(app):
    chart, segs, order = _prepared_chart(app)
    try:
        assert "1/4 validated" in chart.chart.title()
        segs[0].validate()
        chart.refresh_values()
        _wait(app, 50)
        assert "2/4 validated" in chart.chart.title()
    finally:
        chart.close()
