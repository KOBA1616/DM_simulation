from dm_toolkit.gui.editor.transforms.legacy_to_command import convert_legacy_action


def test_convert_action_with_ACTION_key():
    payload = {'ACTION': 'PLAY_CARD', 'card_id': 42}
    out = convert_legacy_action(payload)
    assert out['type'] == 'PLAY_CARD'
    assert out['params']['card_id'] == 42
    assert 'legacy_action' in out


def test_convert_action_with_action_key_and_extra():
    payload = {'action': 'DRAW', 'amount': 2, 'note': 'legacy'}
    out = convert_legacy_action(payload)
    assert out['type'] == 'DRAW'
    assert out['params']['amount'] == 2
    assert out['legacy_action']['note'] == 'legacy'


def test_convert_unknown_payload_wrapped():
    payload = {'unknown_flag': True}
    out = convert_legacy_action(payload)
    assert out['type'] == 'LEGACY_CMD'
    assert 'legacy_action' in out
