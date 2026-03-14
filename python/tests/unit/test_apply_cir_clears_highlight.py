# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.gui.editor.models import CommandModel

class FakeWidget:
    def __init__(self, value):
        self._value = value
        self.last_style = None
    def get_value(self):
        return self._value
    def set_value(self, v):
        self._value = v
    def setStyleSheet(self, s):
        self.last_style = s


def test_apply_cir_clears_highlight():
    form = object.__new__(UnifiedActionForm)
    form.widgets_map = {}
    form.current_model = CommandModel(type='TEST')

    w1 = FakeWidget(1)
    w2 = FakeWidget('a')
    # simulate highlighted state
    w1.setStyleSheet('background: yellow;')
    w2.setStyleSheet('background: yellow;')

    form.widgets_map['amount'] = w1
    form.widgets_map['name'] = w2

    # cir payload will set values (so apply happens)
    cir = [{'type': 'TEST', 'payload': {'amount': 2, 'name': 'b'}}]

    updated = form.apply_cir(cir)
    assert updated is True
    # highlights should be cleared
    assert w1.last_style == '' or w1.last_style is None
    assert w2.last_style == '' or w2.last_style is None
