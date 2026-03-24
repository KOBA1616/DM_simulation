# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.models import CommandModel, QueryParams, TransitionParams, ModifierParams


def test_query_params_are_typed_on_ingest():
    data = {
        'type': 'QUERY',
        'params': {
            'query_string': 'select * from cards',
            'options': ['A', 'B']
        }
    }
    cmd = CommandModel.model_validate(data)
    assert isinstance(cmd.params, QueryParams)
    assert cmd.params.query_string == 'select * from cards'


def test_transition_params_are_typed_on_ingest():
    data = {'type': 'TRANSITION', 'params': {'target_state': 'END_TURN', 'reason': 'test'}}
    cmd = CommandModel.model_validate(data)
    assert isinstance(cmd.params, TransitionParams)
    assert cmd.params.target_state == 'END_TURN'


def test_modifier_params_are_typed_on_ingest():
    data = {'type': 'MODIFY', 'params': {'amount': 3, 'scope': 'ALL'}}
    cmd = CommandModel.model_validate(data)
    assert isinstance(cmd.params, ModifierParams)
    assert cmd.params.amount == 3
