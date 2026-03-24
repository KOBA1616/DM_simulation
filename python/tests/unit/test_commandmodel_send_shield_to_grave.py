from dm_toolkit.gui.editor.models import CommandModel, SendShieldToGraveParams


def test_commandmodel_ingests_send_shield_to_grave_params():
    data = {
        'type': 'SEND_SHIELD_TO_GRAVE',
        'params': {
            'amount': 2,
            'target_group': 'PLAYER_OPPONENT'
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, SendShieldToGraveParams)
    assert cmd.params.amount == 2
    assert cmd.params.target_group == 'PLAYER_OPPONENT'


def test_serialize_flattens_send_shield_to_grave_params():
    data = {'type': 'SEND_SHIELD_TO_GRAVE', 'params': {'amount': 1}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['amount'] == 1
