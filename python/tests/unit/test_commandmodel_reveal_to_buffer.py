from dm_toolkit.gui.editor.models import CommandModel, RevealToBufferParams


def test_commandmodel_ingests_reveal_to_buffer_params():
    data = {
        'type': 'REVEAL_TO_BUFFER',
        'params': {
            'from_zone': 'DECK',
            'amount': 3
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, RevealToBufferParams)
    assert cmd.params.from_zone == 'DECK'
    assert cmd.params.amount == 3


def test_serialize_flattens_reveal_to_buffer_params():
    data = {'type': 'REVEAL_TO_BUFFER', 'params': {'from_zone': 'HAND', 'amount': 2}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['from_zone'] == 'HAND'
    assert out['amount'] == 2
