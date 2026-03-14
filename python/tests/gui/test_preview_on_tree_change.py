# -*- coding: utf-8 -*-
from __future__ import annotations

from dm_toolkit.gui.editor.window import CardEditor


def test_on_tree_changed_triggers_preview_and_selection(monkeypatch):
    # Create an uninitialized CardEditor instance
    win = object.__new__(CardEditor)

    # Fake tree_widget with currentIndex returning a valid-like object
    class _Idx:
        def isValid(self):
            return True

    class FakeTree:
        def currentIndex(self):
            return _Idx()

    called = {}

    # Fake inspector with set_selection that records calls
    class FakeInspector:
        def set_selection(self, idx):
            called['inspector'] = True

    win.tree_widget = FakeTree()
    win.inspector = FakeInspector()

    # Replace update_current_preview with a spy
    def _spy_preview():
        called['preview'] = True

    win.update_current_preview = _spy_preview

    # Call the method under test
    win.on_tree_changed()

    assert called.get('inspector') is True
    assert called.get('preview') is True
