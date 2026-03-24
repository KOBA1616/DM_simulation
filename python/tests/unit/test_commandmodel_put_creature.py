from dm_toolkit.gui.editor.models import CommandModel, PutCreatureParams


def test_commandmodel_ingests_put_creature_params():
    data = {
        'type': 'PUT_CREATURE',
        'params': {
            'card_id': 123,
            'from_zone': 'HAND',
            'to_zone': 'BATTLE',
            'tapped': True
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, PutCreatureParams)
    assert cmd.params.card_id == 123
    assert cmd.params.from_zone == 'HAND'
    assert cmd.params.tapped is True


def test_serialize_flattens_put_creature_params():
    data = {'type': 'PUT_CREATURE', 'params': {'card_id': 7, 'to_zone': 'BATTLE', 'summoned_for_free': False}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['card_id'] == 7
    assert out['to_zone'] == 'BATTLE'
    assert out['summoned_for_free'] is False


def test_put_creature_persists_filter_and_amount_fields():
    data = {
        'type': 'PUT_CREATURE',
        'params': {
            'from_zone': 'HAND',
            'amount': 1,
            'target_group': 'PLAYER_SELF',
            'target_filter': {
                'types': ['ELEMENT'],
                'max_cost': 2,
                'zones': ['HAND'],
            },
        },
    }

    cmd = CommandModel.model_validate(data)
    assert isinstance(cmd.params, PutCreatureParams)
    out = cmd.model_dump()

    assert out['from_zone'] == 'HAND'
    assert out['amount'] == 1
    assert out['target_group'] == 'PLAYER_SELF'
    assert out['target_filter']['types'] == ['ELEMENT']
    assert out['target_filter']['max_cost'] == 2
