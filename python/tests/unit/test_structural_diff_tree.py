# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm

class FakeWidget:
    def __init__(self, value):
        self._value = value
    def get_value(self):
        return self._value


def test_compute_structural_diff_tree_nested():
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

    tree = form.compute_structural_diff_tree(cir_payload)

    # expected nested structure
    assert tree.get('target_filter', {}).get('cost') is True
    assert tree.get('options', {}).get(1, {}).get('label') is True
    assert tree.get('extra') is True
    # unchanged nodes should not be marked
    assert 'type' not in tree.get('target_filter', {})
    assert tree.get('options', {}).get(0) is None or tree.get('options', {}).get(0) == {}
