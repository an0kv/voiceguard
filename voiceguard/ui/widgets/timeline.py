from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPalette
from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class TimeSegment:
    start_sec: float
    end_sec: float


class TimelineWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._times: list[float] = []
        self._values: list[Optional[float]] = []
        self._threshold: Optional[float] = None
        self._segments: list[TimeSegment] = []

        self.setMinimumHeight(220)
        self.setAutoFillBackground(True)

    def clear(self) -> None:
        self._times.clear()
        self._values.clear()
        self._segments.clear()
        self.update()

    def set_data(
        self,
        *,
        times: list[float],
        values: list[Optional[float]],
        threshold: Optional[float] = None,
        segments: Optional[list[TimeSegment]] = None,
    ) -> None:
        self._times = list(times)
        self._values = list(values)
        self._threshold = None if threshold is None else float(threshold)
        self._segments = list(segments or [])
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(12.0, 12.0, -12.0, -18.0)
        base = self.palette().color(QPalette.ColorRole.Base)
        alt = self.palette().color(QPalette.ColorRole.AlternateBase)
        text = self.palette().color(QPalette.ColorRole.Text)
        grad = None
        try:
            from PySide6.QtGui import QLinearGradient  # type: ignore

            grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
            grad.setColorAt(0.0, base)
            grad.setColorAt(1.0, alt)
        except Exception:
            grad = None

        painter.fillRect(rect, grad or base)

        # Axes / grid.
        grid_color = QColor(text)
        grid_color.setAlpha(26)
        grid_pen = QPen(grid_color)
        grid_pen.setWidthF(1.0)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            y = rect.top() + rect.height() * (i / 4.0)
            painter.drawLine(rect.left(), y, rect.right(), y)

        # No data yet.
        if not self._times or not self._values:
            muted = QColor(text)
            muted.setAlpha(140)
            painter.setPen(muted)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Нет данных")
            painter.end()
            return

        t_min = float(min(self._times))
        t_max = float(max(self._times))
        if t_max <= t_min:
            t_max = t_min + 1e-6

        def x_of(t: float) -> float:
            return rect.left() + (float(t) - t_min) / (t_max - t_min) * rect.width()

        def y_of(v: float) -> float:
            v = min(max(float(v), 0.0), 1.0)
            return rect.bottom() - v * rect.height()

        # Alert segments overlay.
        if self._segments:
            seg_color = QColor(249, 115, 22, 46)  # orange-500 alpha
            for seg in self._segments:
                sx = x_of(seg.start_sec)
                ex = x_of(seg.end_sec)
                painter.fillRect(QRectF(sx, rect.top(), max(0.0, ex - sx), rect.height()), seg_color)

        # Threshold line.
        if self._threshold is not None:
            th = float(self._threshold)
            if 0.0 <= th <= 1.0:
                th_pen = QPen(QColor(245, 158, 11, 190))  # amber-500
                th_pen.setStyle(Qt.PenStyle.DashLine)
                th_pen.setWidthF(1.2)
                painter.setPen(th_pen)
                y = y_of(th)
                painter.drawLine(rect.left(), y, rect.right(), y)

        # Series line.
        line_color = QColor(self.palette().color(QPalette.ColorRole.Highlight))
        line_color.setAlpha(220)
        line_pen = QPen(line_color)
        line_pen.setWidthF(2.0)
        painter.setPen(line_pen)

        prev_x: Optional[float] = None
        prev_y: Optional[float] = None
        for t, v in zip(self._times, self._values):
            if v is None:
                prev_x = None
                prev_y = None
                continue
            x = x_of(float(t))
            y = y_of(float(v))
            if prev_x is not None and prev_y is not None:
                painter.drawLine(prev_x, prev_y, x, y)
            prev_x, prev_y = x, y

        # Labels.
        label_color = QColor(text)
        label_color.setAlpha(160)
        painter.setPen(label_color)
        painter.drawText(
            QRectF(rect.left(), rect.bottom() + 2.0, rect.width(), 16.0),
            Qt.AlignmentFlag.AlignLeft,
            f"{t_min:.1f}s",
        )
        painter.drawText(
            QRectF(rect.left(), rect.bottom() + 2.0, rect.width(), 16.0),
            Qt.AlignmentFlag.AlignRight,
            f"{t_max:.1f}s",
        )

        painter.end()
