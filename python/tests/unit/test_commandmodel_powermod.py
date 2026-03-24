from dm_toolkit.gui.editor.models import CommandModel, PowerModParams


def test_commandmodel_ingests_powermod_params():
    data = {
        'type': 'POWER_MOD',
        'params': {
            'amount': 300,
            'target_group': 'PLAYER_SELF'
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, PowerModParams)
    assert cmd.params.amount == 300
    assert cmd.params.target_group == 'PLAYER_SELF'


def test_serialize_flattens_powermod_params():
    data = {'type': 'POWER_MOD', 'params': {'amount': 150, 'target_group': 'PLAYER_OPPONENT'}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['amount'] == 150
    assert out['target_group'] == 'PLAYER_OPPONENT'
