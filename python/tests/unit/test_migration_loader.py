from dm_toolkit.gui.editor.models.serializer import ModelSerializer


def test_migrate_legacy_action_wrapped():
    serializer = ModelSerializer()

    card = {
        'id': 123,
        'name': 'LegacyCard',
        'effects': [
            {
                'trigger': 'ON_PLAY',
                'commands': [
                    {'ACTION': 'PLAY_CARD', 'card_id': 5},
                    {'type': 'DRAW', 'amount': 1}
                ]
            }
        ]
    }

    # Run migration
    serializer._migrate_legacy_card(card)

    cmds = card['effects'][0]['commands']
    assert isinstance(cmds, list)
    legacy_cmd = cmds[0]
    assert legacy_cmd.get('type') == 'PLAY_CARD'
    assert legacy_cmd.get('params', {}).get('card_id') == 5
    # Non-legacy command untouched
    assert cmds[1].get('type') == 'DRAW'
