from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_shield_burn_with_amount():
    action = {'type': 'SHIELD_BURN'}
    out = CardTextGenerator._format_game_action_command('SHIELD_BURN', action, False, 3, 0, '', '', input_key='', input_usage='', sample=[])
    assert '相手のシールドを3つ選び、墓地に置く。' == out


def test_shield_burn_default_amount():
    action = {'type': 'SHIELD_BURN'}
    out = CardTextGenerator._format_game_action_command('SHIELD_BURN', action, False, 0, 0, '', '', input_key='', input_usage='', sample=[])
    assert '相手のシールドを1つ選び、墓地に置く。' == out
