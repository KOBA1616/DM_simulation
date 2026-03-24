# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm

class FakeWidget:
    def __init__(self, value):
        self._value = value
    def get_value(self):
        return self._value


def test_compute_diff_summary_keys():
    form = object.__new__(UnifiedActionForm)
    form.widgets_map = {}
    form.current_model = None

    form.widgets_map['a'] = FakeWidget(1)
    form.widgets_map['b'] = FakeWidget('x')
    form.widgets_map['c'] = FakeWidget(None)

    cir_payload = {'a': 2, 'b': 'x', 'd': 5}

    diff = form.compute_diff_summary(cir_payload)
    # should detect 'a' as changed, 'b' same, 'c' missing, 'd' extra
    assert 'a' in diff
    assert 'b' not in diff
    assert 'd' in diff
