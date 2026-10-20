"""Tests for the PyQt6 dummy-video file selector."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

import sun_find_west_v2.camera.vid_dummy as vid_dummy


def test_select_video_file_uses_qt_dialog_and_retains_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create one Qt application and use its file dialog for video selection."""
    created_apps: list[object] = []
    dialog_calls: list[tuple[object, ...]] = []

    class FakeApplication:
        current: FakeApplication | None = None

        def __init__(self, args: list[str]) -> None:
            self.args = args
            self.quit_on_last_window_closed = True
            created_apps.append(self)
            FakeApplication.current = self

        @classmethod
        def instance(cls) -> FakeApplication | None:
            return cls.current

        def setQuitOnLastWindowClosed(self, enabled: bool) -> None:
            self.quit_on_last_window_closed = enabled

    class FakeFileDialog:
        @staticmethod
        def getOpenFileName(*args: object) -> tuple[str, str]:
            dialog_calls.append(args)
            return (r"C:\videos\sun.ser", "video filter")

    qt_widgets = ModuleType("PyQt6.QtWidgets")
    qt_widgets.QApplication = FakeApplication
    qt_widgets.QFileDialog = FakeFileDialog
    qt_package = ModuleType("PyQt6")
    qt_package.QtWidgets = qt_widgets
    monkeypatch.setitem(sys.modules, "PyQt6", qt_package)
    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", qt_widgets)
    monkeypatch.setattr(vid_dummy, "_qt_application", None)

    selected = vid_dummy._select_video_file()
    selected_again = vid_dummy._select_video_file()

    assert selected == r"C:\videos\sun.ser"
    assert selected_again == r"C:\videos\sun.ser"
    assert len(created_apps) == 1
    assert getattr(created_apps[0], "quit_on_last_window_closed") is False
    assert len(dialog_calls) == 2
    assert "動画ファイル" in str(dialog_calls[0][3])
