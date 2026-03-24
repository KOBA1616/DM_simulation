# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm

class FakeWidget:
    def __init__(self, value):
        self._value = value
    def get_value(self):
        return self._value


def test_compute_structural_diff_dict_and_list():
    form = object.__new__(UnifiedActionForm)
    form.widgets_map = {}
    form.current_model = None

    # top-level dict widget with nested key 'cost' different
    form.widgets_map['target_filter'] = FakeWidget({'cost': 2, 'type': 'A'})
    # list of dicts widget; second element 'label' differs
    form.widgets_map['options'] = FakeWidget([{'label': 1}, {'label': 2}])

    cir_payload = {
        'target_filter': {'cost': 3, 'type': 'A'},
        'options': [{'label': 1}, {'label': 99}],
        'extra': 'value'
    }

    diffs = form.compute_structural_diff(cir_payload)

    assert 'target_filter.cost' in diffs
    assert 'options[1].label' in diffs
    assert 'extra' in diffs
    # unchanged paths should not appear
    assert 'target_filter.type' not in diffs
    assert 'options[0].label' not in diffs
