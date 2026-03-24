from dm_toolkit.gui.editor.models import CommandModel, LookToBufferParams


def test_commandmodel_ingests_look_to_buffer_params():
    data = {
        'type': 'LOOK_TO_BUFFER',
        'params': {
            'from_zone': 'DECK',
            'amount': 3,
            'input_var': 'buf1'
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, LookToBufferParams)
    assert cmd.params.from_zone == 'DECK'
    assert cmd.params.amount == 3
    assert cmd.params.input_var == 'buf1'


def test_serialize_flattens_look_to_buffer_params():
    data = {'type': 'LOOK_TO_BUFFER', 'params': {'from_zone': 'GRAVEYARD', 'amount': 2, 'input_var': 'b'}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['from_zone'] == 'GRAVEYARD'
    assert out['amount'] == 2
    # input_var is nested inside params for LOOK_TO_BUFFER and should be flattened as 'input_var'
    assert out['input_var'] == 'b'
