"""PyQt6 dashboard for live solar-position tracking."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class _ImageLabel(QLabel):
    """Display the camera feed and an overview of the current viewport."""

    def __init__(
        self,
        on_resize: Callable[[], None],
        on_wheel: Callable[[int, QPointF], None],
        on_mouse_press: Callable[[QPointF, bool], None],
        on_mouse_move: Callable[[QPointF, bool], None],
        on_mouse_release: Callable[[], None],
        on_double_click: Callable[[QPointF], None],
    ) -> None:
        """Initialize resize and pointer callbacks."""
        super().__init__()
        self._on_resize = on_resize
        self._on_wheel = on_wheel
        self._on_mouse_press = on_mouse_press
        self._on_mouse_move = on_mouse_move
        self._on_mouse_release = on_mouse_release
        self._on_double_click = on_double_click
        self._minimap_image: QImage | None = None
        self._minimap_view_rect: tuple[float, float, float, float] | None = None
        self._minimap_widget_rect: QRectF | None = None
        self._minimap_image_rect: QRectF | None = None
        self.setMouseTracking(True)

    def set_minimap_data(
        self,
        image: QImage,
        view_rect: tuple[float, float, float, float],
    ) -> None:
        """Update the overview image and its active viewport rectangle."""
        self._minimap_image = image
        self._minimap_view_rect = view_rect
        self.update()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        """Keep the frame fitted to the current label dimensions."""
        super().resizeEvent(a0)
        self._on_resize()

    def wheelEvent(self, a0: QWheelEvent | None) -> None:
        """Delegate the mouse wheel to the zoom controller."""
        if a0 is not None:
            self._on_wheel(a0.angleDelta().y(), a0.position())
            a0.accept()

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        """Delegate a click to the pan or overview-map controller."""
        if a0 is not None:
            self._on_mouse_press(
                a0.position(),
                a0.button() == Qt.MouseButton.LeftButton,
            )
            a0.accept()

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        """Delegate pointer movement while the left button is held."""
        if a0 is not None:
            is_left_down = bool(a0.buttons() & Qt.MouseButton.LeftButton)
            self._on_mouse_move(a0.position(), is_left_down)
            a0.accept()

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        """Stop panning when the mouse button is released."""
        if a0 is not None:
            if a0.button() == Qt.MouseButton.LeftButton:
                self._on_mouse_release()
            a0.accept()

    def mouseDoubleClickEvent(self, a0: QMouseEvent | None) -> None:
        """Delegate a double click to reset the viewport."""
        if a0 is not None:
            self._on_double_click(a0.position())
            a0.accept()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Draw the overview map over the camera feed."""
        super().paintEvent(a0)
        image = self._minimap_image
        if image is None or image.isNull():
            return

        outer = QRectF(12.0, float(self.height()) - 137.0, 174.0, 125.0)
        inner = QRectF(
            outer.left() + 6.0,
            outer.top() + 19.0,
            outer.width() - 12.0,
            outer.height() - 25.0,
        )
        scale = min(inner.width() / image.width(), inner.height() / image.height())
        image_width = image.width() * scale
        image_height = image.height() * scale
        image_rect = QRectF(
            inner.left() + (inner.width() - image_width) / 2.0,
            inner.top() + (inner.height() - image_height) / 2.0,
            image_width,
            image_height,
        )
        self._minimap_widget_rect = outer
        self._minimap_image_rect = image_rect

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#52667e"), 1.0))
        painter.setBrush(QColor(12, 18, 27, 232))
        painter.drawRoundedRect(outer, 7.0, 7.0)
        painter.setPen(QColor("#e6edf6"))
        painter.drawText(QPointF(outer.left() + 8.0, outer.top() + 13.0), "OVERVIEW")
        painter.fillRect(inner, QColor("#080c12"))
        painter.drawImage(image_rect, image)

        view = self._minimap_view_rect
        if view is not None:
            left, top, view_width, view_height = view
            active_rect = QRectF(
                image_rect.left() + left / image.width() * image_rect.width(),
                image_rect.top() + top / image.height() * image_rect.height(),
                view_width / image.width() * image_rect.width(),
                view_height / image.height() * image_rect.height(),
            )
            painter.fillRect(active_rect, QColor(115, 190, 255, 38))
            painter.setPen(QPen(QColor("#9bd2ff"), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(active_rect)
        painter.end()


class _MainWindow(QMainWindow):
    """Notify the visualizer when the main window closes."""

    def __init__(self, on_close: Callable[[], None]) -> None:
        """Initialize the close callback."""
        super().__init__()
        self._on_close = on_close

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Record window closure before delegating to Qt."""
        self._on_close()
        super().closeEvent(a0)


class Visualizer:
    """Render live camera frames and tracking metrics in a PyQt6 window.

    The public methods match the previous Matplotlib visualizer so the
    observation loop and CSV writer can remain unchanged.
    """

    def __init__(
        self,
        width: int,
        height: int,
        acceptable: float = 1,
        grid_color: str = "#FFFFFF",
        grid_ny: int = 2,
        grid_nx: int = 4,
        grid_r: float = 300,
        grid_alpha: float = 0.4,
        opencircle_triangle: bool = False,
    ) -> None:
        """Create the live dashboard and camera controls."""
        del opencircle_triangle
        self.width = width
        self.height = height
        self.acceptable = float(acceptable)
        self.grid_color = grid_color
        self.grid_ny = max(0, int(grid_ny))
        self.grid_nx = max(0, int(grid_nx))
        self.grid_r = max(0.0, float(grid_r))
        self.grid_alpha = min(1.0, max(0.0, float(grid_alpha)))
        self.sunline = min(width, height) / 4
        self._closed = False
        self._latest_image: QImage | None = None
        self._previous_frame_time = time.monotonic()
        self._zoom_factor = 1.0
        self._view_center_x = width / 2.0
        self._view_center_y = height / 2.0
        self._is_panning = False
        self._last_pan_position: QPointF | None = None
        self.buttons: dict[str, QPushButton] = {}
        self.sliders: dict[str, QSlider] = {}
        self._metric_labels: dict[str, QLabel] = {}

        app_instance = QApplication.instance()
        if app_instance is None:
            app = QApplication(sys.argv[:1])
        elif isinstance(app_instance, QApplication):
            app = app_instance
        else:
            raise RuntimeError(
                "SUN_watching requires a QApplication instance, not a QCoreApplication."
            )
        self.app: QApplication = app
        self.app.setApplicationName("SUN_watching")

        self.window = _MainWindow(self._mark_closed)
        self.window.setWindowTitle("SUN_watching | Solar Tracking")
        self.window.setMinimumSize(1040, 720)
        self.window.resize(1440, 900)

        root = QWidget()
        root.setObjectName("root")
        self.window.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(20, 18, 20, 14)
        root_layout.setSpacing(14)

        header = QHBoxLayout()
        title_stack = QVBoxLayout()
        app_title = QLabel("SUN watching")
        app_title.setObjectName("appTitle")
        subtitle = QLabel("Solar tracking and west-angle analysis")
        subtitle.setObjectName("subtitle")
        title_stack.addWidget(app_title)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack)
        header.addStretch(1)
        self.status_pill = QLabel("WAITING FOR FRAME")
        self.status_pill.setObjectName("statusPill")
        header.addWidget(self.status_pill)
        root_layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        root_layout.addLayout(body, 1)

        feed_panel = self._make_panel()
        feed_layout = QVBoxLayout(feed_panel)
        feed_layout.setContentsMargins(14, 14, 14, 14)
        feed_layout.setSpacing(10)
        feed_header = QHBoxLayout()
        feed_title = QLabel("LIVE FEED")
        feed_title.setObjectName("sectionTitle")
        self.frame_label = QLabel("FRAME —")
        self.frame_label.setObjectName("mutedLabel")
        feed_header.addWidget(feed_title)
        feed_header.addStretch(1)
        feed_header.addWidget(self.frame_label)
        feed_layout.addLayout(feed_header)
        self.zoom_label = QLabel("ZOOM 1.0×")
        self.zoom_label.setObjectName("mutedLabel")
        self.image_label = _ImageLabel(
            self._render_image,
            self._handle_zoom_wheel,
            self._handle_mouse_press,
            self._handle_mouse_move,
            self._handle_mouse_release,
            self._handle_double_click,
        )
        self.image_label.setText("Waiting for camera frame")
        self.image_label.setObjectName("imageCanvas")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 470)
        feed_header.addWidget(self.zoom_label)
        feed_layout.addWidget(self.image_label, 1)

        time_legend = QHBoxLayout()
        time_legend.setSpacing(8)
        legend_title = QLabel("TRAJECTORY TIME")
        legend_title.setObjectName("mutedLabel")
        earlier_label = QLabel("EARLIER")
        earlier_label.setObjectName("mutedLabel")
        gradient_bar = QFrame()
        gradient_bar.setObjectName("trajectoryGradient")
        gradient_bar.setMinimumWidth(70)
        gradient_bar.setFixedHeight(8)
        gradient_bar.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #440154, stop:0.25 #3f5b8a, stop:0.5 #21918c, "
            "stop:0.75 #5ec962, stop:1 #fde725); border-radius: 4px;"
        )
        now_label = QLabel("NOW")
        now_label.setObjectName("mutedLabel")
        time_legend.addWidget(legend_title)
        time_legend.addStretch(1)
        time_legend.addWidget(earlier_label)
        time_legend.addWidget(gradient_bar, 1)
        time_legend.addWidget(now_label)
        feed_layout.addLayout(time_legend)
        body.addWidget(feed_panel, 3)

        side_panel = self._make_panel()
        side_panel.setMinimumWidth(270)
        side_panel.setMaximumWidth(360)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(12)

        analysis_title = QLabel("ANALYSIS")
        analysis_title.setObjectName("sectionTitle")
        side_layout.addWidget(analysis_title)

        angle_card = QFrame()
        angle_card.setObjectName("angleCard")
        angle_layout = QVBoxLayout(angle_card)
        angle_layout.setContentsMargins(14, 12, 14, 12)
        angle_caption = QLabel("WEST ANGLE")
        angle_caption.setObjectName("mutedLabel")
        self.angle_label = QLabel("N/A")
        self.angle_label.setObjectName("angleValue")
        self.angle_detail = QLabel("Waiting for sufficient trajectory data")
        self.angle_detail.setObjectName("mutedLabel")
        self.angle_detail.setWordWrap(True)
        angle_layout.addWidget(angle_caption)
        angle_layout.addWidget(self.angle_label)
        angle_layout.addWidget(self.angle_detail)
        side_layout.addWidget(angle_card)

        metric_grid = QGridLayout()
        metric_grid.setHorizontalSpacing(8)
        metric_grid.setVerticalSpacing(8)
        metrics = [
            ("center", "CENTER (X, Y)", "—, —"),
            ("radius", "DETECTED RADIUS", "—"),
            ("frames", "FRAMES", "0"),
            ("points", "TRAJECTORY POINTS", "0"),
            ("fps", "FRAME RATE", "— FPS"),
        ]
        for index, (key, caption, initial) in enumerate(metrics):
            row, column = divmod(index, 2)
            card = QFrame()
            card.setObjectName("metricCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 9, 10, 9)
            caption_label = QLabel(caption)
            caption_label.setObjectName("mutedLabel")
            value_label = QLabel(initial)
            value_label.setObjectName("metricValue")
            value_label.setWordWrap(True)
            card_layout.addWidget(caption_label)
            card_layout.addWidget(value_label)
            metric_grid.addWidget(card, row, column)
            self._metric_labels[key] = value_label
        side_layout.addLayout(metric_grid)

        controls_title = QLabel("CAMERA CONTROLS")
        controls_title.setObjectName("sectionTitle")
        side_layout.addWidget(controls_title)
        self.slider_layout = QVBoxLayout()
        self.slider_layout.setSpacing(10)
        side_layout.addLayout(self.slider_layout)
        side_layout.addStretch(1)
        body.addWidget(side_panel, 1)

        self.action_layout = QHBoxLayout()
        self.action_layout.setSpacing(8)
        self.action_layout.addStretch(1)
        root_layout.addLayout(self.action_layout)
        footer = QLabel(
            "Local observation session   |   Results are written to the report directory"
        )
        footer.setObjectName("footer")
        root_layout.addWidget(footer)

        self.window.setStyleSheet(self._style_sheet())
        self.window.show()
        self.app.processEvents()

    @staticmethod
    def _make_panel() -> QFrame:
        """Create a reusable panel container."""
        panel = QFrame()
        panel.setObjectName("panel")
        return panel

    @staticmethod
    def _style_sheet() -> str:
        """Return the compact dark dashboard theme."""
        return """
            QWidget#root { background: #10151d; color: #e6edf6; }
            QLabel { color: #e6edf6; }
            QLabel#appTitle { font-size: 27px; font-weight: 700; }
            QLabel#subtitle, QLabel#mutedLabel, QLabel#footer { color: #8d9aab; }
            QLabel#subtitle { font-size: 12px; }
            QLabel#footer { font-size: 10px; padding-top: 2px; }
            QLabel#sectionTitle { color: #9eafc5; font-size: 11px; font-weight: 700; }
            QFrame#panel { background: #171f2b; border: 1px solid #293444; border-radius: 12px; }
            QFrame#angleCard { background: #202c3b; border: 1px solid #324357; border-radius: 9px; }
            QFrame#metricCard { background: #1b2532; border: 1px solid #2a3849; border-radius: 8px; }
            QLabel#angleValue { color: #f5f8fc; font-size: 30px; font-weight: 700; }
            QLabel#metricValue { color: #eef4fb; font-size: 15px; font-weight: 600; }
            QLabel#statusPill { background: #263448; color: #b6c9df; border-radius: 10px; padding: 7px 11px; font-size: 10px; font-weight: 700; }
            QLabel#imageCanvas { background: #080c12; border: 1px solid #283445; border-radius: 8px; color: #738299; }
            QPushButton { background: #27364a; color: #edf4fc; border: 1px solid #3b4d63; border-radius: 7px; padding: 8px 14px; font-weight: 600; }
            QPushButton:hover { background: #334760; }
            QPushButton#dangerButton { background: #6d3438; border-color: #955055; }
            QPushButton#dangerButton:hover { background: #844046; }
            QSlider::groove:horizontal { background: #344356; height: 5px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #73b4ff; width: 14px; margin: -5px 0; border-radius: 7px; }
        """

    def _mark_closed(self) -> None:
        """Record that the dashboard is closed."""
        self._closed = True

    def _view_size_for_zoom(self, zoom: float) -> tuple[float, float]:
        """Return the source-image dimensions visible at a zoom level."""
        if zoom <= 1.0:
            return float(self.width), float(self.height)
        label_width = max(1.0, float(self.image_label.width()))
        label_height = max(1.0, float(self.image_label.height()))
        fit_scale = min(label_width / self.width, label_height / self.height)
        return (
            min(float(self.width), label_width / (fit_scale * zoom)),
            min(float(self.height), label_height / (fit_scale * zoom)),
        )

    def _current_view_rect(self) -> tuple[float, float, float, float]:
        """Return the source-image rectangle currently visible in the feed."""
        view_width, view_height = self._view_size_for_zoom(self._zoom_factor)
        left = min(
            max(0.0, self._view_center_x - view_width / 2.0),
            max(0.0, self.width - view_width),
        )
        top = min(
            max(0.0, self._view_center_y - view_height / 2.0),
            max(0.0, self.height - view_height),
        )
        return left, top, view_width, view_height

    def _display_geometry(
        self,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Return the visible image rectangle and its scale/offset in the label."""
        left, top, view_width, view_height = self._current_view_rect()
        label_width = max(1.0, float(self.image_label.width()))
        label_height = max(1.0, float(self.image_label.height()))
        scale = min(label_width / view_width, label_height / view_height)
        offset_x = (label_width - view_width * scale) / 2.0
        offset_y = (label_height - view_height * scale) / 2.0
        return left, top, view_width, view_height, scale, offset_x, offset_y

    def _label_to_image(self, position: QPointF) -> tuple[float, float]:
        """Convert a label position to source-image coordinates."""
        left, top, view_width, view_height, scale, offset_x, offset_y = (
            self._display_geometry()
        )
        x = left + (position.x() - offset_x) / scale
        y = top + (position.y() - offset_y) / scale
        return (
            min(float(self.width), max(0.0, x)),
            min(float(self.height), max(0.0, y)),
        )

    def _clamp_view_center(self) -> None:
        """Keep the viewport inside the source image."""
        view_width, view_height = self._view_size_for_zoom(self._zoom_factor)
        self._view_center_x = min(
            self.width - view_width / 2.0,
            max(view_width / 2.0, self._view_center_x),
        )
        self._view_center_y = min(
            self.height - view_height / 2.0,
            max(view_height / 2.0, self._view_center_y),
        )

    def _handle_zoom_wheel(self, delta: int, position: QPointF) -> None:
        """Zoom around the pointer so the area beneath it remains anchored."""
        if delta == 0 or self._latest_image is None:
            return
        old_left, old_top, old_width, old_height = self._current_view_rect()
        image_x, image_y = self._label_to_image(position)
        fraction_x = min(1.0, max(0.0, (image_x - old_left) / old_width))
        fraction_y = min(1.0, max(0.0, (image_y - old_top) / old_height))
        factor = 1.2 if delta > 0 else 1.0 / 1.2
        self._zoom_factor = min(12.0, max(1.0, self._zoom_factor * factor))
        new_width, new_height = self._view_size_for_zoom(self._zoom_factor)
        self._view_center_x = image_x - (fraction_x - 0.5) * new_width
        self._view_center_y = image_y - (fraction_y - 0.5) * new_height
        self._clamp_view_center()
        self._render_image()

    def _handle_mouse_press(self, position: QPointF, is_left_button: bool) -> None:
        """Center the viewport from the overview map or begin dragging."""
        if not is_left_button:
            return
        map_rect = self.image_label._minimap_widget_rect
        image_rect = self.image_label._minimap_image_rect
        if map_rect is not None and map_rect.contains(position):
            if image_rect is not None and image_rect.contains(position):
                self._view_center_x = (
                    (position.x() - image_rect.left()) / image_rect.width() * self.width
                )
                self._view_center_y = (
                    (position.y() - image_rect.top()) / image_rect.height() * self.height
                )
                self._clamp_view_center()
                self._render_image()
            return
        if self._zoom_factor <= 1.0:
            return
        self._is_panning = True
        self._last_pan_position = position
        self.image_label.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _handle_mouse_move(self, position: QPointF, is_left_down: bool) -> None:
        """Pan the viewport while dragging the camera feed."""
        if not self._is_panning or not is_left_down or self._last_pan_position is None:
            return
        delta_x = position.x() - self._last_pan_position.x()
        delta_y = position.y() - self._last_pan_position.y()
        _, _, _, _, scale, _, _ = self._display_geometry()
        if scale > 0.0:
            self._view_center_x -= delta_x / scale
            self._view_center_y -= delta_y / scale
            self._clamp_view_center()
            self._render_image()
        self._last_pan_position = position

    def _handle_mouse_release(self) -> None:
        """Finish dragging and restore the pointer cursor."""
        self._is_panning = False
        self._last_pan_position = None
        cursor = (
            Qt.CursorShape.OpenHandCursor
            if self._zoom_factor > 1.0
            else Qt.CursorShape.ArrowCursor
        )
        self.image_label.setCursor(cursor)

    def _handle_double_click(self, position: QPointF) -> None:
        """Reset the zoom and return to the full frame."""
        del position
        self._zoom_factor = 1.0
        self._view_center_x = self.width / 2.0
        self._view_center_y = self.height / 2.0
        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
        self._render_image()

    @staticmethod
    def _trajectory_color(progress: float) -> QColor:
        """Map normalized trajectory time to a Viridis-like color."""
        palette = (
            (68, 1, 84),
            (72, 35, 116),
            (64, 67, 135),
            (52, 94, 141),
            (41, 120, 142),
            (32, 144, 140),
            (34, 167, 132),
            (68, 190, 112),
            (121, 209, 81),
            (189, 222, 38),
            (253, 231, 37),
        )
        value = min(1.0, max(0.0, float(progress))) * (len(palette) - 1)
        lower = int(value)
        upper = min(len(palette) - 1, lower + 1)
        fraction = value - lower
        start = palette[lower]
        end = palette[upper]
        channels = tuple(
            round(first + (last - first) * fraction)
            for first, last in zip(start, end, strict=True)
        )
        return QColor(channels[0], channels[1], channels[2])

    @staticmethod
    def _normalise_trajectory_time(
        points: np.ndarray,
        timestamps: np.ndarray | None,
    ) -> np.ndarray:
        """Return a 0–1 chronology axis, falling back to point order."""
        count = len(points)
        if count <= 1:
            return np.zeros(count, dtype=float)
        if timestamps is not None:
            values = np.asarray(timestamps, dtype=float).reshape(-1)
            if values.size == count and np.all(np.isfinite(values)):
                span = float(values.max() - values.min())
                if span > 0.0:
                    return (values - values.min()) / span
        return np.linspace(0.0, 1.0, count)

    @staticmethod
    def _array_to_qimage(frame: np.ndarray) -> QImage:
        """Convert monochrome or RGB pixels into a detached Qt image."""
        pixels = np.asarray(frame)
        if pixels.ndim not in (2, 3):
            raise ValueError(f"Unsupported frame shape: {pixels.shape}")

        if pixels.dtype != np.uint8:
            values = pixels.astype(np.float32)
            low = float(values.min())
            high = float(values.max())
            if high > 255.0:
                span = high - low
                values = (values - low) * (255.0 / span) if span > 0.0 else np.zeros_like(values)
            pixels = np.clip(values, 0, 255).astype(np.uint8)

        pixels = np.ascontiguousarray(pixels)
        if pixels.ndim == 2:
            image_format = QImage.Format.Format_Grayscale8
        elif pixels.shape[2] == 3:
            image_format = QImage.Format.Format_RGB888
        elif pixels.shape[2] == 4:
            image_format = QImage.Format.Format_RGBA8888
        else:
            raise ValueError(f"Unsupported channel count: {pixels.shape[2]}")

        height, width = pixels.shape[:2]
        return QImage(
            pixels.tobytes(),
            width,
            height,
            int(pixels.strides[0]),
            image_format,
        ).copy()

    def add_button(
        self,
        name: str,
        label: str,
        on_clicked: Callable[[], None],
        position: tuple[float, float, float, float] | None = None,
    ) -> QPushButton:
        """Add an action button while preserving the existing visualizer API."""
        del position
        button = QPushButton(label, self.window)
        if name == "quit":
            button.setObjectName("dangerButton")
        button.clicked.connect(lambda _checked=False: on_clicked())
        self.buttons[name] = button
        self.action_layout.addWidget(button)
        return button

    def add_slider(
        self,
        name: str,
        valmin: float,
        valmax: float,
        valinit: float,
        on_change: Callable[[float], None] | None = None,
        label: str | None = None,
        valfmt: str | None = None,
    ) -> QSlider:
        """Add a camera control slider with a numeric value readout."""
        control = QFrame()
        control_layout = QVBoxLayout(control)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)
        header = QHBoxLayout()
        caption = QLabel(label or name.replace("_", " ").title())
        caption.setObjectName("mutedLabel")
        value_label = QLabel(self._format_value(valinit, valfmt))
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(caption, 1)
        header.addWidget(value_label)
        control_layout.addLayout(header)

        slider = QSlider(Qt.Orientation.Horizontal, control)
        minimum = int(round(valmin))
        maximum = max(minimum, int(round(valmax)))
        slider.setRange(minimum, maximum)
        slider.setValue(min(maximum, max(minimum, int(round(valinit)))))
        slider.setPageStep(max(1, (maximum - minimum) // 100))
        slider.valueChanged.connect(
            lambda value: self._slider_value_changed(
                value, value_label, on_change, valfmt
            )
        )
        control_layout.addWidget(slider)
        self.slider_layout.addWidget(control)
        self.sliders[name] = slider
        return slider

    @staticmethod
    def _format_value(value: float, valfmt: str | None) -> str:
        """Format a slider value with an optional printf-style format."""
        if valfmt:
            try:
                return valfmt % value
            except (TypeError, ValueError):
                pass
        return f"{value:g}"

    def _slider_value_changed(
        self,
        value: int,
        value_label: QLabel,
        on_change: Callable[[float], None] | None,
        valfmt: str | None,
    ) -> None:
        """Update the readout and invoke the camera callback."""
        value_label.setText(self._format_value(float(value), valfmt))
        if on_change is not None:
            on_change(float(value))

    def get_slider_val(self, name: str) -> float:
        """Return the current value for a named slider."""
        try:
            return float(self.sliders[name].value())
        except KeyError as error:
            raise KeyError(f"Unknown slider: {name}") from error

    def set_slider_val(self, name: str, value: float) -> None:
        """Set a named slider value."""
        try:
            self.sliders[name].setValue(int(round(value)))
        except KeyError as error:
            raise KeyError(f"Unknown slider: {name}") from error

    def update(
        self,
        img: np.ndarray,
        cx: float,
        cy: float,
        r: float,
        recent_pts: np.ndarray,
        robust_angle: float | None,
        robust_vector: tuple[float, float],
        frame_idx: int | None = None,
        total_frames: int | str | None = None,
        recent_timestamps: np.ndarray | None = None,
    ) -> None:
        """Refresh the camera image, trajectory overlay and metric labels."""
        image = self._array_to_qimage(img).convertToFormat(QImage.Format.Format_RGB888)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        grid_color = QColor(self.grid_color)
        grid_color.setAlpha(round(self.grid_alpha * 255))
        painter.setPen(QPen(grid_color, 2.0))
        for index in range(1, self.grid_ny):
            y = image.height() * index / self.grid_ny
            painter.drawLine(QPointF(0.0, y), QPointF(float(image.width()), y))
        for index in range(1, self.grid_nx):
            x = image.width() * index / self.grid_nx
            painter.drawLine(QPointF(x, 0.0), QPointF(x, float(image.height())))
        if self.grid_r > 0.0:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QPointF(image.width() / 2, image.height() / 2),
                self.grid_r,
                self.grid_r,
            )

        points = np.asarray(recent_pts)

        painter.setPen(QPen(QColor("#91d8ff"), 2.5))
        painter.setBrush(QColor(80, 180, 255, 45))
        painter.drawEllipse(
            QPointF(float(cx), float(cy)), max(0.0, float(r)), max(0.0, float(r))
        )

        west_angle: float | None = None
        if robust_angle is not None:
            west_angle = (180.0 - float(robust_angle) + 180.0) % 360.0 - 180.0
            within_tolerance = abs(west_angle) < self.acceptable
            vector = np.asarray(robust_vector, dtype=float).reshape(-1)
            if vector.size >= 2:
                vector_y, vector_x = float(vector[0]), float(vector[1])
                if float(np.hypot(vector_x, vector_y)) > 0.0:
                    color = "#62e6a7" if within_tolerance else "#d79aff"
                    painter.setPen(QPen(QColor(color), 4.0))
                    painter.drawLine(
                        QPointF(float(cx), float(cy)),
                        QPointF(
                            float(cx) + self.sunline * vector_x,
                            float(cy) + self.sunline * vector_y,
                        ),
                    )

        # Draw the trajectory last and color each segment by its capture time.
        if points.ndim == 2 and points.shape[1] >= 2 and len(points) > 1:
            time_axis = self._normalise_trajectory_time(points, recent_timestamps)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for index, (first, second) in enumerate(
                zip(points[:-1], points[1:], strict=True)
            ):
                progress = float((time_axis[index] + time_axis[index + 1]) / 2.0)
                painter.setPen(QPen(self._trajectory_color(progress), 2.6))
                painter.drawLine(
                    QPointF(float(first[0]), float(first[1])),
                    QPointF(float(second[0]), float(second[1])),
                )
            painter.setPen(QPen(self._trajectory_color(0.0), 1.0))
            painter.setBrush(self._trajectory_color(0.0))
            painter.drawEllipse(QPointF(float(points[0, 0]), float(points[0, 1])), 3.0, 3.0)
            painter.setPen(QPen(self._trajectory_color(1.0), 1.0))
            painter.setBrush(self._trajectory_color(1.0))
            painter.drawEllipse(
                QPointF(float(points[-1, 0]), float(points[-1, 1])), 4.0, 4.0
            )

        painter.end()
        self._latest_image = image
        self._render_image()

        now = time.monotonic()
        elapsed = now - self._previous_frame_time
        self._previous_frame_time = now
        fps = 1.0 / elapsed if elapsed > 0.0 else 0.0
        self._metric_labels["center"].setText(f"{cx:.1f}, {cy:.1f}")
        self._metric_labels["radius"].setText(f"{r:.1f} px")
        count = frame_idx if frame_idx is not None else "—"
        self._metric_labels["frames"].setText(
            f"{count}" + (f" / {total_frames}" if total_frames is not None else "")
        )
        self._metric_labels["points"].setText(str(len(points)))
        self._metric_labels["fps"].setText(f"{fps:.1f} FPS")

        if west_angle is None:
            self.angle_label.setText("N/A")
            self.angle_detail.setText("Waiting for trajectory data")
            self._set_status("INSUFFICIENT DATA", "#263448", "#b6c9df")
        else:
            within_tolerance = abs(west_angle) < self.acceptable
            self.angle_label.setText(f"{west_angle:+.2f}°")
            description = "Within tolerance" if within_tolerance else "Adjustment needed"
            self.angle_detail.setText(f"Target ±{self.acceptable:g}°  ·  {description}")
            if within_tolerance:
                self._set_status("ALIGNED", "#183f35", "#83edbe")
            else:
                self._set_status("ADJUST", "#55352e", "#ffc49e")

        self.frame_label.setText(f"FRAME {count}")
        self.app.processEvents()

    def _set_status(self, text: str, background: str, foreground: str) -> None:
        """Set the status pill colors and label."""
        self.status_pill.setText(text)
        self.status_pill.setStyleSheet(
            f"background: {background}; color: {foreground}; "
            "border-radius: 10px; padding: 7px 11px;"
        )

    def _render_image(self) -> None:
        """Render the current crop and update the overview viewport rectangle."""
        latest_image = self._latest_image
        if latest_image is None:
            return
        left, top, view_width, view_height = self._current_view_rect()
        if self._zoom_factor <= 1.0:
            crop = latest_image
            visible_rect = (
                0.0,
                0.0,
                float(latest_image.width()),
                float(latest_image.height()),
            )
        else:
            x0 = max(0, min(self.width - 1, int(round(left))))
            y0 = max(0, min(self.height - 1, int(round(top))))
            crop_width = max(1, min(self.width - x0, int(round(view_width))))
            crop_height = max(1, min(self.height - y0, int(round(view_height))))
            crop = latest_image.copy(x0, y0, crop_width, crop_height)
            visible_rect = (
                float(x0),
                float(y0),
                float(crop_width),
                float(crop_height),
            )
        scaled = crop.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))
        self.image_label.set_minimap_data(latest_image, visible_rect)
        self.zoom_label.setText(f"ZOOM {self._zoom_factor:.1f}×")
        if self._zoom_factor > 1.0 and not self._is_panning:
            self.image_label.setCursor(Qt.CursorShape.OpenHandCursor)
        elif not self._is_panning:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

    def is_alive(self) -> bool:
        """Return whether the dashboard is still open."""
        return not self._closed and self.window.isVisible()

    def close(self) -> None:
        """Close the dashboard and process pending Qt events."""
        if self.window.isVisible():
            self.window.close()
        self._closed = True
        self.app.processEvents()
