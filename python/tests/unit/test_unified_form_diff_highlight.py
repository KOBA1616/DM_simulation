# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.gui.editor.models import CommandModel


class FakeWidget:
    def __init__(self, value):
        self._value = value
        self.last_style = None

    def get_value(self):
        return self._value

    def setStyleSheet(self, s):
        self.last_style = s


def test_highlight_diff_marks_changed_widgets():
    # Prepare form object without running full __init__ UI
    form = object.__new__(UnifiedActionForm)
    form.widgets_map = {}
    form.current_model = CommandModel(type='TEST')
    form.current_model.params = {'amount': 5, 'optional': False}

    # Widgets reflect current UI values
    w_amount = FakeWidget(5)
    w_optional = FakeWidget(False)
    w_extra = FakeWidget('foo')

    form.widgets_map['amount'] = w_amount
    form.widgets_map['optional'] = w_optional
    form.widgets_map['str_param'] = w_extra

    # CIR payload differs for 'amount' and 'str_param'
    cir_payload = {'amount': 7, 'str_param': 'bar'}

    # Call highlight
    form.highlight_diff(cir_payload)

    # amount should be highlighted (different)
    assert w_amount.last_style is not None and 'background' in w_amount.last_style
    # optional unchanged -> no highlight
    assert w_optional.last_style == '' or w_optional.last_style is None
    # str_param different -> highlighted
    assert w_extra.last_style is not None and 'background' in w_extra.last_style

    # Clear highlights
    form.clear_diff_highlight()
    assert w_amount.last_style == '' or w_amount.last_style is None
    assert w_extra.last_style == '' or w_extra.last_style is None
