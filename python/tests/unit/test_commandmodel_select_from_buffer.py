from dm_toolkit.gui.editor.models import CommandModel, SelectFromBufferParams


def test_commandmodel_ingests_select_from_buffer_params():
    data = {
        'type': 'SELECT_FROM_BUFFER',
        'params': {
            'filter': {'races': ['DRAGON']},
            'amount': -1
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, SelectFromBufferParams)
    assert cmd.params.amount == -1
    assert cmd.params.filter == {'races': ['DRAGON']}


def test_serialize_flattens_select_from_buffer_params():
    data = {'type': 'SELECT_FROM_BUFFER', 'params': {'filter': {'cost': {'min':1}}, 'amount': 2}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['amount'] == 2
    assert out['filter'] == {'cost': {'min':1}}
