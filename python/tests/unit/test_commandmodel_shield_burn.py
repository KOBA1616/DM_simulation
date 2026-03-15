from dm_toolkit.gui.editor.models import CommandModel, ShieldBurnParams


def test_commandmodel_ingests_shield_burn_params():
    data = {
        'type': 'SHIELD_BURN',
        'params': {
            'amount': 2,
            'target_group': 'PLAYER_OPPONENT'
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, ShieldBurnParams)
    assert cmd.params.amount == 2
    assert cmd.params.target_group == 'PLAYER_OPPONENT'


def test_serialize_flattens_shield_burn_params():
    data = {'type': 'SHIELD_BURN', 'params': {'amount': 1}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['amount'] == 1
