# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm

class FakeWidget:
    def __init__(self, value):
        self._value = value
    def get_value(self):
        return self._value


def test_format_structural_diff_multiline():
    form = object.__new__(UnifiedActionForm)
    form.widgets_map = {}
    form.current_model = None

    form.widgets_map['target_filter'] = FakeWidget({'cost': 2, 'type': 'A'})
    form.widgets_map['options'] = FakeWidget([{'label': 1}, {'label': 2}])

    cir_payload = {
        'target_filter': {'cost': 3, 'type': 'A'},
        'options': [{'label': 1}, {'label': 99}],
        'extra': 'value'
    }

    s = form.format_structural_diff(cir_payload)
    # Expect human-readable multiline output containing changed paths
    assert 'target_filter.cost' in s
    assert 'options[1].label' in s
    assert 'extra' in s
    assert '\n' in s  # multiline
