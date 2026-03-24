from dm_toolkit.gui.editor.models import CommandModel, ReplaceCardMoveParams


def test_commandmodel_ingests_replace_card_move_params():
    data = {
        'type': 'REPLACE_CARD_MOVE',
        'params': {
            'from_zone': 'HAND',
            'to_zone': 'BATTLE',
            'replacement_to_zone': 'GRAVEYARD',
            'amount': 2
        }
    }

    cmd = CommandModel.model_validate(data)

    assert isinstance(cmd.params, ReplaceCardMoveParams)
    assert cmd.params.from_zone == 'HAND'
    assert cmd.params.replacement_to_zone == 'GRAVEYARD'


def test_serialize_flattens_replace_card_move_params():
    data = {'type': 'REPLACE_CARD_MOVE', 'params': {'to_zone': 'BATTLE', 'amount': 1}}
    cmd = CommandModel.model_validate(data)
    out = cmd.model_dump()
    assert out['to_zone'] == 'BATTLE'
    assert out['amount'] == 1
