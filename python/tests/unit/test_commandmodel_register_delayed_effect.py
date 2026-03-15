from dm_toolkit.gui.editor.models import CommandModel, RegisterDelayedEffectParams


def test_commandmodel_ingests_register_delayed_effect_params():
    data = {
        'type': 'REGISTER_DELAYED_EFFECT',
        'params': {
            'str_param': 'SOME_EFFECT_ID',
            'amount': 3
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, RegisterDelayedEffectParams)
    assert cmd.params.str_param == 'SOME_EFFECT_ID'
    assert cmd.params.amount == 3


def test_serialize_flattens_register_delayed_effect_params():
    data = {'type': 'REGISTER_DELAYED_EFFECT', 'params': {'str_param': 'EID', 'amount': 2}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['str_param'] == 'EID'
    assert out['amount'] == 2
