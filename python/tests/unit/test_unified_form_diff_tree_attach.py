# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.gui.editor.forms.diff_tree_widget import DiffTreeWidget

class FakeWidget:
    def __init__(self, value):
        self._value = value
    def get_value(self):
        return self._value

class FakeItem:
    def __init__(self, cir):
        self._cir = cir
    def data(self, role):
        if role == 'ROLE_CIR':
            return self._cir
        return None


def test_unified_form_attaches_diff_tree():
    form = object.__new__(UnifiedActionForm)
    # provide minimal attributes used by attach logic
    form.widgets_map = {}
    form.diff_tree_widget = DiffTreeWidget()

    # setup widgets to compare
    form.widgets_map['target_filter'] = FakeWidget({'cost': 2, 'type': 'A'})
    form.widgets_map['options'] = FakeWidget([{'label': 1}, {'label': 2}])

    cir_payload = {'type':'X','payload':{'target_filter': {'cost': 3, 'type': 'A'}, 'options': [{'label':1}, {'label':99}], 'extra':'value'}}
    item = FakeItem([cir_payload])

    # simulate attachment: compute structural diff tree and set it on the widget
    tree = form.compute_structural_diff_tree(cir_payload['payload'])
    form.diff_tree_widget.set_diff_tree(tree)

    lines = form.diff_tree_widget.get_lines()
    assert 'target_filter.cost' in lines
    assert 'options[1].label' in lines
    assert 'extra' in lines
