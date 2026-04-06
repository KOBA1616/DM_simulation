from dm_toolkit.gui.editor.models import CommandModel, ReplaceCardMoveParams


def test_commandmodel_ingests_replace_card_move_params():
    data = {
        'type': 'REPLACE_CARD_MOVE',
        'params': {
            'target_group': 'PLAYER_OPPONENT',
            'from_zone': 'HAND',
            'to_zone': 'BATTLE',
            'replacement_to_zone': 'GRAVEYARD',
            'up_to': True,
            'optional': False,
            'target_filter': {'types': ['CREATURE'], 'max_power': {'input_link': 'card_ref', 'input_value_usage': 'MAX_POWER'}},
            'amount': 2
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, ReplaceCardMoveParams)
    assert cmd.params.target_group == 'PLAYER_OPPONENT'
    assert cmd.params.from_zone == 'HAND'
    assert cmd.params.replacement_to_zone == 'GRAVEYARD'
    assert cmd.params.up_to is True
    assert cmd.params.optional is False
    assert cmd.params.target_filter == {'types': ['CREATURE'], 'max_power': {'input_link': 'card_ref', 'input_value_usage': 'MAX_POWER'}}
    assert cmd.params.filter == cmd.params.target_filter


def test_serialize_flattens_replace_card_move_params():
    data = {
        'type': 'REPLACE_CARD_MOVE',
        'params': {
            'from_zone': 'GRAVEYARD',
            'to_zone': 'DECK_BOTTOM',
            'target_group': 'SELF',
            'target_filter': {'zones': ['BATTLE_ZONE'], 'max_power': 3000},
            'amount': 1,
        }
    }
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['from_zone'] == 'GRAVEYARD'
    assert out['to_zone'] == 'DECK_BOTTOM'
    assert out['amount'] == 1
    assert out['target_group'] == 'SELF'
    assert out['target_filter'] == {'zones': ['BATTLE_ZONE'], 'max_power': 3000}
    assert out['filter'] == {'zones': ['BATTLE_ZONE'], 'max_power': 3000}
