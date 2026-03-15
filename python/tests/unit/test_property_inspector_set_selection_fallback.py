# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.property_inspector import PropertyInspector


class _FakeLabel:
    def __init__(self):
        self.visible = True
        self.text = ""

    def setText(self, text):
        self.text = text

    def setVisible(self, visible):
        self.visible = visible

    def setToolTip(self, _text):
        return None


class _FakeStack:
    def __init__(self):
        self.current = None

    def setCurrentWidget(self, widget):
        self.current = widget


class _FakeWidget:
    def __init__(self, should_raise=False):
        self.should_raise = should_raise
        self.set_data_called = False

    def set_data(self, _item):
        self.set_data_called = True
        if self.should_raise:
            raise RuntimeError("boom")


class _FakeItem:
    def data(self, _role):
        return None


class _FakeModel:
    def itemFromIndex(self, _index):
        return _FakeItem()


class _FakeIndex:
    def __init__(self, item_type):
        self._item_type = item_type

    def isValid(self):
        return True

    def data(self, _role):
        return self._item_type

    def model(self):
        return _FakeModel()


class _FakeInvalidIndex:
    def isValid(self):
        return False


def test_set_selection_falls_back_to_empty_page_when_form_set_data_raises():
    inspector = object.__new__(PropertyInspector)
    inspector._update_breadcrumb = lambda _idx: None
    inspector.stack = _FakeStack()
    inspector.empty_page = object()
    inspector.cir_label = _FakeLabel()

    bad_widget = _FakeWidget(should_raise=True)
    inspector.form_map = {"CARD": bad_widget}

    inspector.set_selection(_FakeIndex("CARD"))

    assert bad_widget.set_data_called is True
    assert inspector.stack.current is inspector.empty_page


def test_set_selection_invalid_index_shows_empty_page():
    inspector = object.__new__(PropertyInspector)
    inspector._update_breadcrumb = lambda _idx: None
    inspector.stack = _FakeStack()
    inspector.empty_page = object()
    inspector.cir_label = _FakeLabel()
    inspector.form_map = {}

    inspector.set_selection(_FakeInvalidIndex())

    assert inspector.stack.current is inspector.empty_page
