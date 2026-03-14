import pytest

from dm_toolkit.gui.editor.models import CommandModel, TransitionParams, ModifierParams


def test_commandmodel_ingests_transition_params():
    data = {
        'type': 'TRANSITION',
        'params': {
            'target_state': 'COMPLETED',
            'reason': 'auto'
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, TransitionParams)
    assert cmd.params.target_state == 'COMPLETED'
    assert cmd.params.reason == 'auto'


def test_commandmodel_ingests_modifier_params():
    data = {
        'type': 'MODIFY',
        'params': {
            'amount': 5,
            'scope': 'HAND'
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, ModifierParams)
    assert cmd.params.amount == 5
    assert cmd.params.scope == 'HAND'


def test_serialize_flattens_typed_params():
    tdata = {'type': 'TRANSITION', 'params': {'target_state': 'X'}}
    mdata = {'type': 'MODIFY', 'params': {'amount': 1}}

    tcmd = CommandModel.model_validate(tdata)
    mcmd = CommandModel.model_validate(mdata)

    td = tcmd.model_dump()
    md = mcmd.model_dump()

    assert td['target_state'] == 'X'
    assert md['amount'] == 1
