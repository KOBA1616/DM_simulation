from dm_toolkit.gui.editor.models import CommandModel, SummonTokenParams


def test_commandmodel_ingests_summon_token_params():
    data = {
        'type': 'SUMMON_TOKEN',
        'params': {
            'token_id': 'DRAGON_TOKEN',
            'amount': 3
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, SummonTokenParams)
    assert cmd.params.token_id == 'DRAGON_TOKEN'
    assert cmd.params.amount == 3


def test_serialize_flattens_summon_token_params():
    data = {'type': 'SUMMON_TOKEN', 'params': {'token_id': 'DRAGON', 'amount': 2}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['token_id'] == 'DRAGON'
    assert out['amount'] == 2
