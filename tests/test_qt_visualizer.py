"""Smoke tests for PyQt6 image conversion."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage

from sun_find_west_v2.qt_visualizer import Visualizer


def test_array_to_qimage_accepts_grayscale_frames() -> None:
    """Convert a grayscale frame and preserve its dimensions."""
    frame = np.zeros((5, 7), dtype=np.uint8)

    image = Visualizer._array_to_qimage(frame)

    assert image.width() == 7
    assert image.height() == 5
    assert image.format() == QImage.Format.Format_Grayscale8


def test_array_to_qimage_accepts_rgb_frames() -> None:
    """Convert an RGB frame and preserve its dimensions."""
    frame = np.zeros((5, 7, 3), dtype=np.uint8)

    image = Visualizer._array_to_qimage(frame)

    assert image.width() == 7
    assert image.height() == 5
    assert image.format() == QImage.Format.Format_RGB888


def test_trajectory_colormap_covers_early_to_late_colors() -> None:
    """Map the start and end of a trajectory to distinct endpoint colors."""
    assert Visualizer._trajectory_color(0.0).name() == "#440154"
    assert Visualizer._trajectory_color(1.0).name() == "#fde725"


def test_trajectory_time_uses_capture_timestamps() -> None:
    """Normalize capture times so irregular frame intervals remain meaningful."""
    points = np.zeros((3, 2))
    normalized = Visualizer._normalise_trajectory_time(
        points,
        np.array([10.0, 15.0, 30.0]),
    )

    assert np.allclose(normalized, [0.0, 0.25, 1.0])


def test_trajectory_time_falls_back_to_point_order() -> None:
    """Use point order if capture timestamps are unavailable."""
    points = np.zeros((3, 2))

    normalized = Visualizer._normalise_trajectory_time(points, None)

    assert np.allclose(normalized, [0.0, 0.5, 1.0])
