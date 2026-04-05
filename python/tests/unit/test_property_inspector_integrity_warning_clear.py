# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.property_inspector import PropertyInspector
from dm_toolkit.gui.i18n import tr


class _FakeLabel:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


def test_integrity_warnings_empty_clears_header_label():
    inspector = object.__new__(PropertyInspector)
    inspector.header_label = _FakeLabel()
    emitted = []

    class _Emitter:
        def emit(self, command, data):
            emitted.append((command, data))

    inspector.structure_update_requested = _Emitter()

    inspector._on_structure_update("INTEGRITY_WARNINGS", {"warnings": ["warn"]})
    assert "warn" in inspector.header_label.text

    inspector._on_structure_update("INTEGRITY_WARNINGS", {"warnings": []})
    assert inspector.header_label.text == tr("Property Inspector")
    assert emitted[-1][0] == "INTEGRITY_WARNINGS"
