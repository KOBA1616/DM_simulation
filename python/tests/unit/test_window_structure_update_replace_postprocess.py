# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.consts import STRUCT_CMD_REPLACE_WITH_COMMAND
from dm_toolkit.gui.editor.window import CardEditor


class _Idx:
    def isValid(self):
        return True


class _Model:
    def __init__(self, selected_item):
        self._selected_item = selected_item

    def itemFromIndex(self, _idx):
        return self._selected_item


class _Item:
    def __init__(self, item_type, parent=None):
        self._item_type = item_type
        self._parent = parent

    def data(self, _role=None):
        return self._item_type

    def parent(self):
        return self._parent

    def index(self):
        return _Idx()


def test_on_structure_update_replace_with_command_uses_centralized_postprocess_once():
    win = CardEditor.__new__(CardEditor)

    card_item = _Item("CARD")
    effect_item = _Item("EFFECT", parent=card_item)
    command_item = _Item("COMMAND", parent=effect_item)

    calls = {"replace": 0, "selection": 0, "preview": 0}

    class _Tree:
        def __init__(self):
            self.standard_model = _Model(command_item)

        def currentIndex(self):
            return _Idx()

        def replace_item_with_command(self, idx, data):
            calls["replace"] += 1
            calls["replaced_data"] = data

    class _Inspector:
        def set_selection(self, _idx):
            calls["selection"] += 1

    def _spy_preview(immediate=False):
        calls["preview"] += 1
        calls["preview_immediate"] = immediate

    win.tree_widget = _Tree()
    win.inspector = _Inspector()
    win.request_preview_update = _spy_preview

    payload = {"target_item": command_item, "new_data": {"type": "FOO"}}
    win.on_structure_update(STRUCT_CMD_REPLACE_WITH_COMMAND, payload)

    assert calls["replace"] == 1
    assert calls["replaced_data"]["type"] == "FOO"
    assert calls["selection"] == 1
    assert calls["preview"] == 1
    assert calls["preview_immediate"] is True
