from dm_toolkit.gui.editor.models import CommandModel, MoveBufferToZoneParams


def test_commandmodel_ingests_move_buffer_to_zone_params():
    data = {
        'type': 'MOVE_BUFFER_TO_ZONE',
        'params': {
            'to_zone': 'GRAVEYARD',
            'amount': 2,
            'filter': {'cost': {'max':3}}
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, MoveBufferToZoneParams)
    assert cmd.params.to_zone == 'GRAVEYARD'
    assert cmd.params.amount == 2
    assert cmd.params.filter == {'cost': {'max':3}}


def test_serialize_flattens_move_buffer_to_zone_params():
    data = {'type': 'MOVE_BUFFER_TO_ZONE', 'params': {'to_zone': 'HAND', 'amount': 1}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['to_zone'] == 'HAND'
    assert out['amount'] == 1
