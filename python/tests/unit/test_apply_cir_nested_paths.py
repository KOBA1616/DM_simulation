# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.gui.editor.models import CommandModel


class DummyListWidget:
    def __init__(self, value):
        self._value = value
    def get_value(self):
        return self._value
    def set_value(self, v):
        self._value = v


def test_apply_cir_updates_nested_list_index():
    # create instance without running Qt init
    form = object.__new__(UnifiedActionForm)
    form.widgets_map = {}
    form.current_model = CommandModel.construct(type='TEST', params={})

    # existing options list with one entry
    init_options = [{'label': 'a'}]
    lw = DummyListWidget(init_options.copy())
    form.widgets_map['options'] = lw

    cir = [{'type': 'TEST', 'payload': {'options[1].label': 'new-label'}}]

    updated = form.apply_cir(cir)

    assert updated is True
    assert 'options' in form.current_model.params
    assert isinstance(form.current_model.params['options'], list)
    assert len(form.current_model.params['options']) >= 2
    assert form.current_model.params['options'][1]['label'] == 'new-label'
    # widget should be updated (merged) as well
    assert lw._value[1]['label'] == 'new-label'


def test_apply_cir_initializes_nested_dict_path():
    form = object.__new__(UnifiedActionForm)
    form.widgets_map = {}
    form.current_model = CommandModel.construct(type='TEST', params={})

    cir = [{'type': 'TEST', 'payload': {'target_filter.cost.min': 3}}]

    updated = form.apply_cir(cir)

    assert updated is True
    assert 'target_filter' in form.current_model.params
    tf = form.current_model.params['target_filter']
    assert isinstance(tf, dict)
    assert 'cost' in tf
    assert isinstance(tf['cost'], dict)
    assert tf['cost']['min'] == 3
