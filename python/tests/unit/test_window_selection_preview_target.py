# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.window import CardEditor


class _Idx:
    def __init__(self, name: str, valid: bool = True):
        self.name = name
        self._valid = valid

    def isValid(self):
        return self._valid


class _Item:
    def __init__(self, item_type: str, parent=None):
        self._item_type = item_type
        self._parent = parent

    def data(self, _role=None):
        return self._item_type

    def parent(self):
        return self._parent


class _CardModel:
    def __init__(self, data):
        self._data = data

    def model_dump(self, by_alias=True):
        return dict(self._data)


class _Selected:
    def __init__(self, indexes):
        self._indexes = indexes

    def indexes(self):
        return list(self._indexes)


class _Timer:
    def stop(self):
        return None

    def start(self):
        return None


def test_selection_change_uses_selected_index_for_preview_when_current_index_is_stale():
    win = object.__new__(CardEditor)
    old_index = _Idx("old")
    new_index = _Idx("new")
    old_item = _Item("CARD")
    new_item = _Item("CARD")

    class _StandardModel:
        def itemFromIndex(self, idx):
            return new_item if idx is new_index else old_item

    class _DataManager:
        def reconstruct_card_model(self, item):
            if item is new_item:
                return _CardModel({"name": "New Card"})
            return _CardModel({"name": "Old Card"})

    class _Tree:
        def __init__(self):
            self.standard_model = _StandardModel()
            self.data_manager = _DataManager()
            self.expanded = []

        def currentIndex(self):
            return old_index

        def expand(self, idx):
            self.expanded.append(idx)

    class _Inspector:
        def __init__(self):
            self.selected = None

        def set_selection(self, idx):
            self.selected = idx

    class _Preview:
        def __init__(self):
            self.rendered = None
            self.cleared = False

        def render_card(self, data):
            self.rendered = data

        def clear_preview(self):
            self.cleared = True

    win.tree_widget = _Tree()
    win.inspector = _Inspector()
    win.preview_widget = _Preview()
    win._preview_debounce_timer = _Timer()
    win._preview_target_index = None

    win.on_selection_changed(_Selected([new_index]), _Selected([]))

    assert win.inspector.selected is new_index
    assert win.preview_widget.rendered["name"] == "New Card"
    assert win.tree_widget.expanded == [new_index]