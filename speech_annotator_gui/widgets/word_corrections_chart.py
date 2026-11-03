"""Bar chart showing percentage of words corrected per segment by confidence order."""

from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSet,
    QChart,
    QChartView,
    QStackedBarSeries,
    QValueAxis,
)
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

_COLOR_VALIDATED = QColor(76, 175, 80)
_COLOR_SKIPPED = QColor(100, 150, 255)
_COLOR_ARROW = QColor(0, 102, 255)
# Bar height (%) used for unvalidated segments. Kept below 100 so the
# blue downward arrow drawn above the bar fits inside the plot area.
_SENTINEL = 88.0
_RECT_COLOR = QColor(255, 0, 0)
_BAR_PIXEL_WIDTH = 50
_CHART_HEIGHT = 250
_Y_AXIS_MARGIN = 80  # pixels reserved for Y axis labels
_ARROW_WIDTH = 14
_ARROW_HEIGHT = 11
_ARROW_GAP = 3  # pixels between the arrow tip and the top of the bar
_CLICK_MARGIN_ABOVE = 20  # clickable pixels above the plot area (arrows)
_TITLE_BASE = "Words Corrected % (by Confidence Order)"


class _ChartView(QChartView):
    """QChartView that overlays the current-segment highlight rectangle
    and blue arrows above unvalidated bars.

    Clicking a bar (or the arrow above it) emits segment_clicked with the
    0-based position of the segment in the chart order.
    """

    segment_clicked = Signal(int)

    def __init__(self, chart, parent=None) -> None:
        super().__init__(chart, parent)
        self._rect: QRectF | None = None
        self._arrows: list[tuple[float, float]] = []
        self._click_area: QRectF | None = None
        self._n_categories = 0
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def set_highlight_rect(self, rect: QRectF | None) -> None:
        self._rect = rect
        self.viewport().update()

    def set_arrows(self, arrows: list[tuple[float, float]]) -> None:
        self._arrows = arrows
        self.viewport().update()

    def set_click_area(self, area: QRectF | None, n_categories: int) -> None:
        self._click_area = area
        self._n_categories = n_categories

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for center_x, bar_top in self._arrows:
            self._draw_arrow(painter, center_x, bar_top)
        if self._rect is not None:
            painter.setPen(QPen(_RECT_COLOR, 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._rect)
        painter.end()

    def _draw_arrow(self, painter: QPainter, center_x: float, bar_top: float) -> None:
        """Draw a blue downward arrow whose tip points at the bar top."""
        tip_y = bar_top - _ARROW_GAP
        polygon = QPolygonF([
            QPointF(center_x, tip_y),
            QPointF(center_x - _ARROW_WIDTH / 2, tip_y - _ARROW_HEIGHT),
            QPointF(center_x + _ARROW_WIDTH / 2, tip_y - _ARROW_HEIGHT),
        ])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_COLOR_ARROW))
        painter.drawPolygon(polygon)

    def _category_at(self, pos: QPointF) -> int | None:
        """Return the category index at the given position, if any."""
        if self._click_area is None or self._n_categories == 0:
            return None
        if not self._click_area.contains(pos):
            return None
        ratio = (pos.x() - self._click_area.x()) / self._click_area.width()
        index = int(ratio * self._n_categories)
        if 0 <= index < self._n_categories:
            return index
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            index = self._category_at(event.position())
            if index is not None:
                self.segment_clicked.emit(index)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._category_at(event.position()) is not None:
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.viewport().unsetCursor()
        super().mouseMoveEvent(event)


class WordCorrectionsChart(QWidget):
    """Bar chart: % words corrected per segment, confidence order.

    Validated segments show green bars with their correction percentage.
    Unvalidated segments show blue bars with a blue downward arrow above
    the bar pointing at it, making unvalidated segments immediately
    visible. The current segment is highlighted with a red rectangle.
    The chart title shows the validated/total progress, and a legend
    explains the bar colors.

    Clicking a bar (or its arrow) emits segment_clicked with the 0-based
    position of that segment in the chart order so the main window can
    navigate to it.

    The chart has a fixed height; when there are more segments than fit
    in the viewport, a horizontal scroll bar appears. The scroll bar is
    automatically centered on the current segment when navigating or
    loading a new file.
    """

    segment_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)

        self.chart = QChart()
        self.chart.setTheme(QChart.ChartTheme.ChartThemeLight)
        self.chart.setTitle(_TITLE_BASE)
        self.chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

        self.chart_view = _ChartView(self.chart)
        self.chart_view.segment_clicked.connect(self.segment_clicked)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setFixedHeight(_CHART_HEIGHT)

        self._scroll = QScrollArea()
        self._scroll.setWidget(self.chart_view)
        self._scroll.setWidgetResizable(False)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFixedHeight(_CHART_HEIGHT + 20)  # +20 for scrollbar
        layout.addWidget(self._scroll)

        self._segments: list = []
        self._order: list[int] = []
        self._current_pos = -1
        self._axis_x: QBarCategoryAxis | None = None
        self._axis_y: QValueAxis | None = None
        self._series: QStackedBarSeries | None = None
        self._bar_validated: QBarSet | None = None
        self._bar_skipped: QBarSet | None = None
        self._apply_scheduled = False
        self._apply_retries = 0
        # Timer parented to this widget: it is destroyed together with the
        # chart, so a pending apply can never fire on a deleted widget.
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self._apply)

    def update_chart(
        self,
        segments: list,
        order: list[int],
        current_pos: int = -1,
    ) -> None:
        """Rebuild the chart with segment data.

        All segments are shown in confidence order. Validated segments
        show their correction percentage (green). Unvalidated segments
        show a blue bar with a blue downward arrow above it. The current
        segment (current_pos, 0-based) is highlighted with a red
        rectangle and centered in the scroll area.

        Args:
            segments: list of SegmentData objects
            order: indices of segments ordered by confidence ascending
            current_pos: 0-based position in the confidence order of the
                current segment, or -1 if none
        """
        self._segments = segments
        self._order = list(order)
        self._current_pos = current_pos

        self.chart.removeAllSeries()
        for axis in self.chart.axes():
            self.chart.removeAxis(axis)
        self.chart_view.set_highlight_rect(None)
        self.chart_view.set_arrows([])
        self.chart_view.set_click_area(None, 0)
        self._series = None
        self._bar_validated = None
        self._bar_skipped = None
        self._axis_x = None
        self._axis_y = None

        n = len(self._order)
        self._update_title()
        if n == 0:
            axis_x = QBarCategoryAxis()
            axis_x.setTitleText("Segment (confidence order)")
            axis_x.append(["—"])
            self.chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)

            axis_y = QValueAxis()
            axis_y.setTitleText("% Words Corrected")
            axis_y.setLabelFormat("%.0f%%")
            axis_y.setMin(0)
            axis_y.setMax(100.0)
            self.chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            self.chart_view.setFixedWidth(300)
            self.chart.setGeometry(0, 0, 300, _CHART_HEIGHT)
            return

        categories = [str(i + 1) for i in range(n)]

        # 2 bar sets: green for validated, blue for unvalidated.
        # Each has n values; only one set per category is non-zero.
        self._bar_validated = QBarSet("Validated")
        self._bar_skipped = QBarSet("Not validated")
        for _ in range(n):
            self._bar_validated.append(0.0)
            self._bar_skipped.append(0.0)
        self._bar_validated.setColor(_COLOR_VALIDATED)
        self._bar_skipped.setColor(_COLOR_SKIPPED)

        self._series = QStackedBarSeries()
        self._series.append(self._bar_validated)
        self._series.append(self._bar_skipped)
        self.chart.addSeries(self._series)

        self._axis_x = QBarCategoryAxis()
        self._axis_x.setTitleText("Segment (confidence order)")
        self._axis_x.append(categories)
        self.chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        self._series.attachAxis(self._axis_x)

        self._axis_y = QValueAxis()
        self._axis_y.setTitleText("% Words Corrected")
        self._axis_y.setLabelFormat("%.0f%%")
        self._axis_y.setMin(0)
        self.chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._series.attachAxis(self._axis_y)

        self._fill_bars()

        # Size the chart view so all bars get _BAR_PIXEL_WIDTH pixels.
        chart_width = n * _BAR_PIXEL_WIDTH + _Y_AXIS_MARGIN
        self.chart_view.setFixedWidth(chart_width)
        self.chart.setGeometry(0, 0, chart_width, _CHART_HEIGHT)

        self._schedule_apply()

    def refresh_values(self) -> None:
        """Recompute bar values in place without rebuilding the chart.

        Called after a text edit or a validation: updates bar heights,
        arrows, axis range and title without recreating the series (much
        faster than a full update_chart for large files).
        """
        if self._series is None or not self._order:
            return
        self._fill_bars()
        self._update_title()
        self._schedule_apply()

    def set_current_position(self, pos: int) -> None:
        """Move the highlight rectangle and scroll to a segment without
        rebuilding the chart."""
        if not self._order or not (0 <= pos < len(self._order)):
            return
        self._current_pos = pos
        self._schedule_apply()

    def _fill_bars(self) -> None:
        """Fill the two bar sets from current segment states."""
        max_pct = 0.0
        for position, idx in enumerate(self._order):
            seg = self._segments[idx]
            if seg.is_validated:
                total_words = len(seg.baseline_text.split())
                if total_words > 0:
                    pct = 100.0 * seg.word_edit_count / total_words
                else:
                    pct = 0.0
                self._bar_validated.replace(position, float(pct))
                self._bar_skipped.replace(position, 0.0)
                if pct > max_pct:
                    max_pct = pct
            else:
                self._bar_validated.replace(position, 0.0)
                self._bar_skipped.replace(position, _SENTINEL)
        self._axis_y.setMax(max(100.0, max_pct + 5))

    def _update_title(self) -> None:
        if not self._segments:
            self.chart.setTitle(_TITLE_BASE)
            return
        n_validated = sum(1 for s in self._segments if s.is_validated)
        self.chart.setTitle(
            f"{_TITLE_BASE} - {n_validated}/{len(self._segments)} validated")

    def _schedule_apply(self) -> None:
        """Defer overlay computation to the next event loop tick so the
        chart has time to lay out its plot area after geometry changes."""
        if self._apply_scheduled:
            return
        self._apply_scheduled = True
        self._apply_timer.start(0)

    def _apply(self) -> None:
        """Compute the highlight rectangle, the arrows above unvalidated
        bars and the scroll position after the chart has finished laying
        out."""
        self._apply_scheduled = False
        if self._series is None or not self._order:
            return
        n = len(self._order)
        plot = self.chart.plotArea()
        if plot.width() <= 0 or plot.height() <= 0:
            # Chart not laid out yet — retry a few times.
            self._apply_retries += 1
            if self._apply_retries <= 10:
                self._apply_scheduled = True
                self._apply_timer.start(25)
            return
        self._apply_retries = 0
        slot_width = plot.width() / n

        # Clickable area: the plot area extended upward to cover arrows.
        click_area = QRectF(
            plot.x(), plot.y() - _CLICK_MARGIN_ABOVE,
            plot.width(), plot.height() + _CLICK_MARGIN_ABOVE)
        self.chart_view.set_click_area(click_area, n)

        # Blue arrows above every unvalidated bar.
        y_sentinel = self._y_for_value(_SENTINEL)
        arrows: list[tuple[float, float]] = []
        for position, idx in enumerate(self._order):
            if not self._segments[idx].is_validated:
                center_x = plot.x() + (position + 0.5) * slot_width
                arrows.append((center_x, y_sentinel))
        self.chart_view.set_arrows(arrows)

        # Red rectangle around the current bar + scroll centering.
        if 0 <= self._current_pos < n:
            bar_width = slot_width * self._series.barWidth()
            center_x = plot.x() + (self._current_pos + 0.5) * slot_width
            self.chart_view.set_highlight_rect(QRectF(
                center_x - bar_width / 2,
                plot.y(),
                bar_width,
                plot.height(),
            ))
            bar_center_x = _Y_AXIS_MARGIN + self._current_pos * _BAR_PIXEL_WIDTH \
                + _BAR_PIXEL_WIDTH / 2
            viewport_width = self._scroll.viewport().width()
            h_bar = self._scroll.horizontalScrollBar()
            # Center the bar in the viewport.
            target = int(bar_center_x) - viewport_width // 2
            h_bar.setValue(max(0, target))
        else:
            self.chart_view.set_highlight_rect(None)

    def _y_for_value(self, value: float) -> float:
        """Y pixel coordinate (chart-local) of a value on the Y axis."""
        plot = self.chart.plotArea()
        y_min = self._axis_y.min()
        y_max = self._axis_y.max()
        if y_max <= y_min:
            return plot.y()
        ratio = (value - y_min) / (y_max - y_min)
        return plot.bottom() - ratio * plot.height()
