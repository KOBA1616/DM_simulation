from dm_toolkit.gui.editor.action_converter import ActionConverter


def test_friend_burst_conversion():
    act = {'type': 'FRIEND_BURST', 'str_val': 'FRIENDX', 'value1': 1, 'scope': 'PLAYER_SELF'}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'FRIEND_BURST'
    assert cmd.get('str_val') == 'FRIENDX'
    assert cmd.get('value1') == 1
    assert cmd.get('target_group') == 'PLAYER_SELF'


def test_cast_spell_conversion():
    act = {'type': 'CAST_SPELL', 'scope': 'PLAYER_OPPONENT', 'optional': True, 'str_val': 'CASTX'}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'CAST_SPELL'
    assert cmd.get('str_val') == 'CASTX'
    assert 'OPTIONAL' in (cmd.get('flags') or [])
    assert cmd.get('target_group') == 'PLAYER_OPPONENT'
