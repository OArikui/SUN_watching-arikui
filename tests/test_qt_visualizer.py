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
