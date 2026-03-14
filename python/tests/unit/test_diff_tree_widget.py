# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.diff_tree_widget import DiffTreeWidget


def test_diff_tree_widget_lines():
    widget = DiffTreeWidget()
    # tree format: nested dicts, ints for list indices
    tree = {
        'target_filter': {'cost': True, 'type': False},
        'options': {1: {'label': True}},
        'extra': True
    }
    widget.set_diff_tree(tree)
    lines = widget.get_lines()
    assert 'target_filter.cost' in lines
    assert 'options[1].label' in lines
    assert 'extra' in lines
    # ensure unchanged/False nodes are not present
    assert 'target_filter.type' not in lines
